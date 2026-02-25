# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from transformers import AutoTokenizer

from config import Config
from model import Transformer
from data_utils import load_data, collate_fn


def train_epoch(model, loader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    pbar = tqdm(loader, desc="Training")

    for src, tgt in pbar:
        src = src.to(Config.DEVICE)
        tgt = tgt.to(Config.DEVICE)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()
        output = model(src, tgt_input)

        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        tgt_output = tgt_output.contiguous().view(-1)

        loss = criterion(output, tgt_output)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    return epoch_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(Config.DEVICE)
            tgt = tgt.to(Config.DEVICE)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            output = model(src, tgt_input)
            output_dim = output.shape[-1]

            loss = criterion(
                output.contiguous().view(-1, output_dim),
                tgt_output.contiguous().view(-1)
            )

            epoch_loss += loss.item()

    return epoch_loss / len(loader)


def main():
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)

    train_ds, valid_ds, vocab_size, pad_idx = load_data(Config.DATA_DIR)

    if len(train_ds) == 0 or len(valid_ds) == 0:
        raise ValueError("資料集為空")

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=Config.BATCH_SIZE,
        collate_fn=collate_fn
    )

    print(f"Tokenizer 詞表大小: {vocab_size}")
    print(f"PAD token id: {pad_idx}")

    model = Transformer(
        vocab_size,
        vocab_size,
        d_model=Config.d_model,
        n_heads=Config.n_heads,
        n_layers=Config.n_layers,
        d_ff=Config.d_ff,
        src_pad_idx=pad_idx,
        tgt_pad_idx=pad_idx,
        dropout=Config.dropout
    ).to(Config.DEVICE)

    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_idx,
        label_smoothing=Config.LABEL_SMOOTH
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LR,
        betas=(0.9, 0.98),
        eps=1e-9
    )

    best_loss = float("inf")
    print("開始訓練")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            Config.GRAD_CLIP
        )

        valid_loss = evaluate(
            model,
            valid_loader,
            criterion
        )

        end_time = time.time()
        mins, secs = divmod(end_time - start_time, 60)

        print(f"Epoch {epoch + 1:02}")
        print(f"Time: {int(mins)}m {int(secs)}s")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "vocab_size": vocab_size,
                        "pad_idx": pad_idx,
                        "d_model": Config.d_model,
                        "n_heads": Config.n_heads,
                        "n_layers": Config.n_layers,
                        "d_ff": Config.d_ff,
                        "dropout": Config.dropout,
                        "tokenizer_name": Config.TOKENIZER_NAME
                    }
                },
                Config.SAVE_PATH
            )
            print(f"模型儲存至 {Config.SAVE_PATH}")


if __name__ == "__main__":
    main()
