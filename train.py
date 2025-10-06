# train.py
import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from data import build_data_loaders, r_parallel, tokenize_en, tokenize_zh
from model import TfMod, masks
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# ---------- 設定 ----------
DR = "data"
NE = 30
BZ = 32
D_MODEL = 512
N_HEADS = 8
N_LAYERS = 6
D_FF = 2048
WS = 4000
DEVICE = torch.device("cuda")
save_path = "transformer_model.pt"
# -------------------------

def get_lr(d_mod, w):
    def lr_step(step):
        step = max(step, 1)
        return (d_mod ** -0.5) * min(step ** -0.5, step * (w ** -1.5))
    return lr_step
# optimizer 優化器
# criterion 損失函數
# scheduler 學習率
# src_pad_idx 輸入序列
# tgt_pad_idx 輸出
def t_epoch(mod, it, optimizer, criterion, scheduler,src_pad_idx, tgt_pad_idx, device,to=100):
    #訓練一個 epoch,並定期印出中途過程 (loss, avg_loss, lr, elapsed)。
    #- src_pad_idx, tgt_pad_idx, device: 用在 generate_masks 與 loss(ignore_index)
    #- to: 每多少 step 印一次

    mod.train()
    epoch_loss = 0.0
    count = 0
    start_time = time.time()

    for i, (src_batch, tgt_batch) in enumerate(it, 1):
        src = src_batch.to(device)   # (batch, src_len)
        tgt = tgt_batch.to(device)   # (batch, tgt_len)

        optimizer.zero_grad()

        # prepare decoder input (remove last token) and target (remove first <sos>)
        input_tgt = tgt[:, :-1]
        target = tgt[:, 1:]
        # 產生 mask
        src_mask, tgt_mask = masks(src, input_tgt, src_pad_idx, tgt_pad_idx, device=device)

        # forward
        output = mod(src, input_tgt, src_mask=src_mask, tgt_mask=tgt_mask)  # (batch, tgt_len-1, vocab)
        o_dim = output.size(-1)

        # 計算 loss（flatten 後計算 token-level cross entropy）
        loss = criterion(output.contiguous().view(-1, o_dim), target.contiguous().view(-1))

        # backward & step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            if type(scheduler).__name__ != "ReduceLROnPlateau":
                scheduler.step()

        batch_loss = loss.item()
        epoch_loss += batch_loss
        count += 1

        if (i % to) == 0:
            elapsed = time.time() - start_time
            avg_loss = epoch_loss / count
            # 取 optimizer 的第一個 param_group 的 lr
            lr = None
            try:
                lr = optimizer.param_groups[0].get('lr', None)
            except Exception:
                lr = None
            lr_str = f"{lr:.2e}" if (lr is not None) else "N/A"
            print(f"[Step {i}] batch_loss={batch_loss:.4f}  avg_loss={avg_loss:.4f}  lr={lr_str}  elapsed={elapsed:.1f}s")

    #回傳平均 loss，輸出summary
    epoch_avg = (epoch_loss / count) if count > 0 else 0.0
    total_time = time.time() - start_time
    print(f"Epoch finished. avg_loss={epoch_avg:.4f}  elapsed={total_time:.1f}s")
    return epoch_avg

def evaluate(mod, it, criterion, src_pad_idx, tgt_pad_idx, device):
    mod.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src_batch, tgt_batch in it:
            src = src_batch.to(device)
            tgt = tgt_batch.to(device)
            input_tgt = tgt[:, :-1]
            target = tgt[:, 1:]
            src_mask, tgt_mask = masks(src, input_tgt, src_pad_idx, tgt_pad_idx, device=device)
            output = mod(src, input_tgt, src_mask=src_mask, tgt_mask=tgt_mask)
            output_dim = output.size(-1)
            loss = criterion(output.contiguous().view(-1, output_dim), target.contiguous().view(-1))
            epoch_loss += loss.item()
    return epoch_loss / len(it)

def load_model_for_inference(path=save_path, device=DEVICE):
    ckpt = torch.load(path, map_location=device)
    src_vocab = ckpt["src_vocab"]
    tgt_vocab = ckpt["tgt_vocab"]
    model_state = ckpt["model_state"]
    model = TfMod(len(src_vocab["itos"]), len(tgt_vocab["itos"])).to(device)
    model.load_state_dict(model_state)
    # set padding_idx to match vocab
    src_pad_idx = src_vocab["stoi"].get("<pad>", 0)
    tgt_pad_idx = tgt_vocab["stoi"].get("<pad>", 0)
    model.src_emb.padding_idx = src_pad_idx
    model.tgt_emb.padding_idx = tgt_pad_idx
    model.eval()
    return model, src_vocab, tgt_vocab

def translate_sentence(model, sentence, src_vocab, tgt_vocab, max_len=50, device=DEVICE):
    model.eval()
    tokens = ["<sos>"] + [tok.text.lower() for tok in tokenize_en(sentence)] + ["<eos>"]
    src_ids = [src_vocab["stoi"].get(tok, src_vocab["stoi"]["<unk>"]) for tok in tokens]
    src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(device)  # (1, src_len)
    # encoder
    src_mask, _ = masks(src_tensor, src_tensor, src_vocab["stoi"]["<pad>"], tgt_vocab["stoi"]["<pad>"], device=device)
    with torch.no_grad():
        enc_out = model.src_emb(src_tensor) * math.sqrt(model.d_model)
        enc_out = model.pos_enc(enc_out)
        for layer in model.enc_layers:
            enc_out = layer(enc_out, src_mask)

    # decode step by step
    trg_ids = [tgt_vocab["stoi"]["<sos>"]]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_ids).unsqueeze(0).to(device)
        _, tgt_mask = masks(src_tensor, trg_tensor, src_vocab["stoi"]["<pad>"], tgt_vocab["stoi"]["<pad>"], device=device)
        dec_in = model.tgt_emb(trg_tensor) * math.sqrt(model.d_model)
        dec_in = model.pos_enc(dec_in)
        dec_out = dec_in
        for layer in model.dec_layers:
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)
        output = model.fc_out(dec_out)  # (1, cur_len, vocab)
        next_token = output.argmax(-1)[:, -1].item()
        trg_ids.append(next_token)
        if next_token == tgt_vocab["stoi"]["<eos>"]:
            break
    return [tgt_vocab["itos"][i] for i in trg_ids]


def get_cm(mod, dataloader, src_vocab, tgt_vocab, device, src_pad_idx, tgt_pad_idx, top_k=40):
    #token-level confusion matrix (teacher-forcing)
    #參數:
    #  - mod: model
    #  - dataloader: validation/test loader
    #  - src_vocab, tgt_vocab: vocab dicts (含 itos/stoi)
    #  - src_pad_idx, tgt_pad_idx: 正確的 pad index (從 vocab 取得)

    mod.eval()
    inv_itos = tgt_vocab["itos"]

    # 選出 top_k tokens（或全部）
    common_tokens = inv_itos[:top_k] if len(inv_itos) >= top_k else inv_itos
    other_label = "<other>"
    labels = list(common_tokens) + [other_label]
    label_to_idx = {tok: i for i, tok in enumerate(labels)}

    all_true = []
    all_pred = []

    with torch.no_grad():
        for src_batch, tgt_batch in dataloader:
            src = src_batch.to(device)   # (B, src_len)
            tgt = tgt_batch.to(device)   # (B, tgt_len)
            input_tgt = tgt[:, :-1]
            target = tgt[:, 1:]  # (B, tgt_len-1)

            # **這裡使用正確的 pad idx**
            src_mask, tgt_mask = masks(src, input_tgt, src_pad_idx, tgt_pad_idx, device=device)
            output = mod(src, input_tgt, src_mask=src_mask, tgt_mask=tgt_mask)  # (B, L, V)
            pred_ids = output.argmax(-1).cpu().numpy()  # (B, L)
            tgt_np = target.cpu().numpy()               # (B, L)

            B, L = tgt_np.shape
            for b in range(B):
                for pos in range(L):
                    t = int(tgt_np[b, pos])
                    if t == tgt_pad_idx:
                        continue
                    p = int(pred_ids[b, pos])
                    true_tok = inv_itos[t] if t < len(inv_itos) else "<unk>"
                    pred_tok = inv_itos[p] if p < len(inv_itos) else "<unk>"
                    all_true.append(label_to_idx.get(true_tok, label_to_idx[other_label]))
                    all_pred.append(label_to_idx.get(pred_tok, label_to_idx[other_label]))

    if len(all_true) == 0:
        cm = np.zeros((len(labels), len(labels)), dtype=int)
    else:
        cm = confusion_matrix(all_true, all_pred, labels=list(range(len(labels))))
    return cm, labels

def plot_confusion_matrix(cm, labels, figsize=(10,8)):
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest')
    plt.title("Token-level Confusion Matrix (top tokens)")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.tight_layout()
    plt.show()


def main():
    #準備資料
    train_loader, valid_loader, test_loader, src_vocab, tgt_vocab = build_data_loaders(data_dir=DR, batch_size=BZ)
    src_pad_idx = src_vocab["stoi"]["<pad>"]
    tgt_pad_idx = tgt_vocab["stoi"]["<pad>"]
    src_vocab_size = len(src_vocab["itos"])
    tgt_vocab_size = len(tgt_vocab["itos"])

    #模型
    model = TfMod(src_vocab_size, tgt_vocab_size, d_model=D_MODEL, num_heads=N_HEADS, num_layers=N_LAYERS, d_ff=D_FF).to(DEVICE)
    model.src_emb.padding_idx = src_pad_idx
    model.tgt_emb.padding_idx = tgt_pad_idx

    #loss/optimizer/scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    lr_lambda = get_lr(D_MODEL, WS)
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    print("src_pad_idx:", src_pad_idx)
    print("tgt_pad_idx:", tgt_pad_idx)
    print("model.src_emb.padding_idx:", model.src_emb.padding_idx)
    print("model.tgt_emb.padding_idx:", model.tgt_emb.padding_idx)
    #訓練
    train_losses, valid_losses = [], []
    best_valid = float("inf")
    for epoch in range(NE):
        start_time = time.time()
        train_loss = t_epoch(model, train_loader, optimizer, criterion, scheduler, src_pad_idx, tgt_pad_idx, DEVICE)
        valid_loss = evaluate(model, valid_loader, criterion, src_pad_idx, tgt_pad_idx, DEVICE)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        end_time = time.time()
        if valid_loss < best_valid:
            best_valid = valid_loss
            torch.save({
                "model_state": model.state_dict(),
                "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab
            }, save_path)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f} | Time: {end_time-start_time:.1f}s")
    cm, labels = get_cm(model, valid_loader, src_vocab, tgt_vocab, DEVICE, src_pad_idx, tgt_pad_idx, top_k=40)
    plot_confusion_matrix(cm, labels)
    #損失曲線
    plt.plot(train_losses, label="train")
    plt.plot(valid_losses, label="valid")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
