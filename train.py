# train.py
import math
import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from data import build_data_loaders, tokenize_en
from model import TfMod, masks
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import sacrebleu
import os
import torch.nn.functional as F
# ---------- 設定 ----------
DR = "data"
NE = 50
BZ = 32
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 4
D_FF = 1024
WS = 4000
DEVICE = torch.device("cuda")
save_path = "transformer_model.pt"
LABEL_SMOOTH = 0.1
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
def t_epoch(mod, it, optimizer, criterion, scheduler, src_pad_idx, tgt_pad_idx, device, to=100, scheduled_sampling_prob=0.0):
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

        # Scheduled Sampling: 隨機選擇使用模型預測或真實標籤作為輸入
        if scheduled_sampling_prob > 0.0 and torch.rand(1).item() < scheduled_sampling_prob:
            with torch.no_grad():
                output = mod(src, input_tgt, src_mask=None, tgt_mask=None)
                input_tgt = output.argmax(-1)  # 使用模型預測作為下一步輸入

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

def inference(path=save_path, device=DEVICE):
    ckpt = torch.load(path, map_location=device)
    src_vocab = ckpt["src_vocab"]
    tgt_vocab = ckpt["tgt_vocab"]
    model_state = ckpt["model_state"]
    model = TfMod(len(src_vocab["itos"]), len(tgt_vocab["itos"]),
                  d_model=D_MODEL, num_heads=N_HEADS, num_layers=N_LAYERS, d_ff=D_FF).to(device)
    model.load_state_dict(model_state)
    src_pad_idx = src_vocab["stoi"].get("<pad>", 0)
    tgt_pad_idx = tgt_vocab["stoi"].get("<pad>", 0)
    model.src_emb.padding_idx = src_pad_idx
    model.tgt_emb.padding_idx = tgt_pad_idx
    model.eval()
    return model, src_vocab, tgt_vocab

def test(model, sentence, src_vocab, tgt_vocab, max_len=50, device=DEVICE):
    model.eval()
    # tokenize_en 已回傳 string token list，因此直接使用；並把 token 轉小寫以一致
    tokens = ["<sos>"] + [tok.lower() for tok in tokenize_en(sentence)] + ["<eos>"]
    src_ids = [src_vocab["stoi"].get(tok, src_vocab["stoi"].get("<unk>", 1)) for tok in tokens]
    src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(device)  # (1, src_len)
    # encoder: 用 dummy trg 產生正確的 src_mask
    dummy_trg = torch.LongTensor([[tgt_vocab["stoi"].get("<sos>", 1)]]).to(device)
    src_mask, _ = masks(src_tensor, dummy_trg, src_vocab["stoi"].get("<pad>",0), tgt_vocab["stoi"].get("<pad>",0), device=device)
    with torch.no_grad():
        enc_out = model.src_emb(src_tensor) * math.sqrt(model.d_model)
        enc_out = model.pos_enc(enc_out)
        for layer in model.enc_layers:
            enc_out = layer(enc_out, src_mask)

    # decode step by step
    trg_ids = [tgt_vocab["stoi"].get("<sos>", 1)]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_ids).unsqueeze(0).to(device)
        _, tgt_mask = masks(src_tensor, trg_tensor, src_vocab["stoi"].get("<pad>",0), tgt_vocab["stoi"].get("<pad>",0), device=device)
        dec_in = model.tgt_emb(trg_tensor) * math.sqrt(model.d_model)
        dec_in = model.pos_enc(dec_in)
        dec_out = dec_in
        for layer in model.dec_layers:
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)
        output = model.fc_out(dec_out)  # (1, cur_len, vocab)
        next_token = output.argmax(-1)[:, -1].item()
        trg_ids.append(next_token)
        if next_token == tgt_vocab["stoi"].get("<eos>", -1):
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

def generate_and_write_hyps_refs(model, dataloader, src_vocab, tgt_vocab, device, out_hyp="hyps.txt", out_ref="refs.txt", max_len=80):
    model.eval()
    hyps = []
    refs = []
    # dataloader yields (src_batch, tgt_batch)
    with torch.no_grad():
        for src_batch, tgt_batch in tqdm(dataloader, desc="Generate"):
            B = src_batch.size(0)
            # 對每個 sample 用 translate_sentence (簡單方式)，可慢但穩定
            for i in range(B):
                # 構造 src_text（依你的 tokenizer 恢復）
                src_ids = src_batch[i].cpu().tolist()
                # 移除 pad
                src_ids = [x for x in src_ids if x != src_vocab["stoi"]["<pad>"]]
                # 轉回 token 字串
                # 若 src 原先是英文 tokenized by words/regex，拼接時用空格
                src_tokens = [src_vocab["itos"][idx] for idx in src_ids]
                src_text = " ".join([tok for tok in src_tokens if tok not in ("<sos>","<eos>","<pad>")])
                pred_tokens = test(model, src_text, src_vocab, tgt_vocab, max_len=max_len, device=device)
                # 轉為句子（中文通常直接 join）
                pred_sent = "".join([t for t in pred_tokens if t not in ("<sos>","<eos>","<pad>")])
                # 參考答案
                tgt_ids = tgt_batch[i].cpu().tolist()
                ref_tokens = [tgt_vocab["itos"][idx] for idx in tgt_ids if idx not in (tgt_vocab["stoi"]["<pad>"], tgt_vocab["stoi"]["<sos>"], tgt_vocab["stoi"]["<eos>"])]
                ref_sent = "".join(ref_tokens)
                hyps.append(pred_sent.strip())
                refs.append(ref_sent.strip())
    # 寫檔
    with open(out_hyp, "w", encoding="utf-8") as fh:
        for s in hyps: fh.write(s + "\n")
    with open(out_ref, "w", encoding="utf-8") as fr:
        for s in refs: fr.write(s + "\n")
    print(f"Saved {out_hyp} and {out_ref} (total {len(hyps)})")
    return out_hyp, out_ref

def compute_bleu_with_sacrebleu(hyp_file, ref_file):

    with open(hyp_file, "r", encoding="utf-8") as fh:
        hyps = [l.strip() for l in fh]
    with open(ref_file, "r", encoding="utf-8") as fr:
        refs = [l.strip() for l in fr]
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    print("SacreBLEU:", bleu.score)
    return bleu.score
def sample_and_print_examples(model, dataloader, src_vocab, tgt_vocab, device, n=20, max_len=80):
    model.eval()
    printed = 0
    with torch.no_grad():
        for src_batch, tgt_batch in dataloader:
            B = src_batch.size(0)
            for i in range(B):
                if printed >= n:
                    return
                # src -> text
                src_ids = src_batch[i].cpu().tolist()
                src_ids = [x for x in src_ids if x != src_vocab["stoi"]["<pad>"]]
                src_tokens = [src_vocab["itos"][idx] for idx in src_ids]
                src_text = " ".join([tok for tok in src_tokens if tok not in ("<sos>","<eos>","<pad>")])
                pred_tokens = test(model, src_text, src_vocab, tgt_vocab, max_len=max_len, device=device)
                pred_sent = "".join([t for t in pred_tokens if t not in ("<sos>","<eos>","<pad>")])
                tgt_ids = tgt_batch[i].cpu().tolist()
                ref_tokens = [tgt_vocab["itos"][idx] for idx in tgt_ids if idx not in (tgt_vocab["stoi"]["<pad>"], tgt_vocab["stoi"]["<sos>"], tgt_vocab["stoi"]["<eos>"])]
                ref_sent = "".join(ref_tokens)
                print("=== Example", printed+1, "===")
                print("SRC:", src_text)
                print("REF:", ref_sent)
                print("PRED:", pred_sent)
                print()
                printed += 1
    print("Done samples.")



def custom_loss(output, target, pad_idx, unk_idx, penalty=0.5):
    # output: (batch, seq_len, vocab_size)
    # target: (batch, seq_len)
    # pad_idx: padding token index
    # unk_idx: <unk> token index
    # penalty: penalty factor for <unk>

    vocab_size = output.size(-1)
    output = output.view(-1, vocab_size)  # (batch * seq_len, vocab_size)
    target = target.view(-1)  # (batch * seq_len)

    # Create a weight tensor for the loss
    weight = torch.ones(vocab_size, device=output.device)
    weight[unk_idx] = penalty  # Apply penalty to <unk>

    # Compute the loss with the custom weight
    loss = F.cross_entropy(output, target, weight=weight, ignore_index=pad_idx)
    return loss

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
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx, label_smoothing=LABEL_SMOOTH)
    optimizer = AdamW(model.parameters(), betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0)

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

    max_samples = 500
    if os.path.exists(save_path):
        print("載入 best checkpoint 供評估...")
        eval_model, src_vocab_ckpt, tgt_vocab_ckpt = inference(path=save_path, device=DEVICE)
        eval_src_vocab, eval_tgt_vocab = src_vocab_ckpt, tgt_vocab_ckpt
    else:
        print("未找到 checkpoint，使用目前記憶體中的 model 供評估。")
        eval_model = model
        eval_src_vocab, eval_tgt_vocab = src_vocab, tgt_vocab
    def generate_and_write_hyps_refs_with_limit(model, dataloader, src_vocab, tgt_vocab, device, out_hyp="hyps.txt", out_ref="refs.txt", max_len=80, max_samples=None):
        model.eval()
        hyps = []
        refs = []
        count = 0
        with torch.no_grad():
            for src_batch, tgt_batch in tqdm(dataloader, desc="Generate"):
                B = src_batch.size(0)
                for i in range(B):
                    if max_samples is not None and count >= max_samples:
                        break
                    src_ids = src_batch[i].cpu().tolist()
                    src_ids = [x for x in src_ids if x != src_vocab["stoi"].get("<pad>",0)]
                    src_tokens = [src_vocab["itos"][idx] for idx in src_ids]
                    src_text = " ".join([tok for tok in src_tokens if tok not in ("<sos>","<eos>","<pad>")])
                    pred_tokens = test(eval_model, src_text, src_vocab, tgt_vocab, max_len=max_len, device=device)
                    pred_sent = "".join([t for t in pred_tokens if t not in ("<sos>","<eos>","<pad>")])
                    tgt_ids = tgt_batch[i].cpu().tolist()
                    ref_tokens = [tgt_vocab["itos"][idx] for idx in tgt_ids if idx not in (tgt_vocab["stoi"].get("<pad>",0), tgt_vocab["stoi"].get("<sos>",1), tgt_vocab["stoi"].get("<eos>",2))]
                    ref_sent = "".join(ref_tokens)
                    hyps.append(pred_sent.strip())
                    refs.append(ref_sent.strip())
                    count += 1
                if max_samples is not None and count >= max_samples:
                    break
        with open("hyps.txt", "w", encoding="utf-8") as fh:
            for s in hyps: fh.write(s + "\n")
        with open("refs.txt", "w", encoding="utf-8") as fr:
            for s in refs: fr.write(s + "\n")
        print(f"Saved hyps.txt and refs.txt (total {len(hyps)})")
        return "hyps.txt", "refs.txt"

    hyp_file, ref_file = generate_and_write_hyps_refs_with_limit(eval_model, valid_loader, eval_src_vocab, eval_tgt_vocab, DEVICE, max_len=80, max_samples=max_samples)
    try:
        bleu_score = compute_bleu_with_sacrebleu(hyp_file, ref_file)
    except Exception as e:
        print("計算 BLEU 發生錯誤或缺少 sacrebleu:", e)
        bleu_score = None

    print("\n--- 抽樣 檢視 ---")
    sample_and_print_examples(eval_model, valid_loader, eval_src_vocab, eval_tgt_vocab, DEVICE, n=20, max_len=80)
    #損失曲線
    plt.plot(train_losses, label="train")
    plt.plot(valid_losses, label="valid")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

