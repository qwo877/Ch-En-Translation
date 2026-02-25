# data_utils.py
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from config import Config
import os

print(f"載入 Tokenizer: {Config.TOKENIZER_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)

class TransDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]
        
        src_enc = tokenizer(src_text, truncation=True, max_length=Config.MAX_LEN, return_tensors='pt')
        tgt_enc = tokenizer(tgt_text, truncation=True, max_length=Config.MAX_LEN, return_tensors='pt')
        
        return src_enc['input_ids'].squeeze(0), tgt_enc['input_ids'].squeeze(0)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    
    return src_padded, tgt_padded

def load_data(data_dir):
    def read_pairs(split):
        path_en = os.path.join(data_dir, f"{split}.en")
        path_zh = os.path.join(data_dir, f"{split}.zh")
        if not os.path.exists(path_en): return []
        with open(path_en, 'r', encoding='utf-8') as fe, open(path_zh, 'r', encoding='utf-8') as fz:
            return [(e.strip(), z.strip()) for e, z in zip(fe, fz) if e.strip() and z.strip()]

    train_pairs = read_pairs("train")
    valid_pairs = read_pairs("valid")
    
    print(f"訓練集: {len(train_pairs)}, 驗證集: {len(valid_pairs)}")
    
    train_ds = TransDataset(train_pairs)
    valid_ds = TransDataset(valid_pairs)
    
    vocab_size = tokenizer.vocab_size
    pad_idx = tokenizer.pad_token_id
    
    return train_ds, valid_ds, vocab_size, pad_idx