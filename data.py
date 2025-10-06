# data.py
import os
from collections import Counter
from typing import List, Tuple, Iterable
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import jieba
try:
    import spacy
    spacy_en = spacy.load("en_core_web_sm")
except Exception:
    spacy_en = None

# --- 設定 ---
DIR = "data"
MIN_F = 2
BZ = 64
# ------------------------------

SPECIALS = ["<pad>", "<unk>", "<sos>", "<eos>"]


def r_parallel(data_dir: str, split: str) -> List[Tuple[str, str]]:
    # 預期 data_dir/ 有 train.en, train.zh, valid.en, valid.zh, test.en, test.zh
    # 回傳 list of (src_sentence, tgt_sentence)
    src_path = os.path.join(data_dir, f"{split}.en")
    tgt_path = os.path.join(data_dir, f"{split}.zh")
    if not os.path.exists(src_path) or not os.path.exists(tgt_path):
        raise FileNotFoundError(f"找不到資料檔: {src_path} 或 {tgt_path}")
    pairs = []
    with open(src_path, "r", encoding="utf-8") as fsrc, open(tgt_path, "r", encoding="utf-8") as ftgt:
        for s_line, t_line in zip(fsrc, ftgt):
            s = s_line.strip()
            t = t_line.strip()
            if s and t:
                pairs.append((s, t))
    return pairs


def tokenize_en(text: str):
    if spacy_en is not None:
        return [tok.text.lower() for tok in spacy_en(text)]
    else:
        import re
        # 把單字與標點分開
        tokens = re.findall(r"\w+|[^\s\w]", text.lower())
        return tokens


def tokenize_zh(text: str):
    return [tok for tok in jieba.cut(text.strip(), cut_all=False) if tok.strip()]


def yield_tokens(pairs: Iterable[Tuple[str, str]], tokenizer, which: str):
    #生成器：從 (src, tgt) pairs yield token sequence(指定哪一側)
    if which == "src":
        for s, _ in pairs:
            yield tokenizer(s)
    else:
        for _, t in pairs:
            yield tokenizer(t)


def build_vocab(tokens_iter: Iterable[List[str]], min_freq=2, specials=SPECIALS):
    counter = Counter()
    for toks in tokens_iter:
        counter.update(toks)
    # keep tokens with freq >= min_freq
    itos = list(specials) + [w for w, c in counter.items() if c >= min_freq and w not in specials]
    stoi = {tok: idx for idx, tok in enumerate(itos)}
    # ensure unk index exists
    if "<unk>" not in stoi:
        stoi["<unk>"] = len(stoi)
        itos.append("<unk>")
    return {"itos": itos, "stoi": stoi}


class ParallelDataset(Dataset):
    #Dataset 回傳已 tokenized 並轉為 id(含 <sos>/<eos>)
    #__getitem__ => (src_ids_tensor, tgt_ids_tensor)


    def __init__(self, pairs: List[Tuple[str, str]], src_vocab: dict, tgt_vocab: dict,
                 src_tokenizer=tokenize_en, tgt_tokenizer=tokenize_zh):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.sos = "<sos>"
        self.eos = "<eos>"
        self.unk = "<unk>"

    def __len__(self):
        return len(self.pairs)

    def _numericalize(self, tokens: List[str], vocab: dict):
        stoi = vocab["stoi"]
        ids = [stoi.get(tok, stoi[self.unk]) for tok in tokens]
        return ids

    def __getitem__(self, idx):
        s, t = self.pairs[idx]
        s_tokens = [self.sos] + self.src_tokenizer(s) + [self.eos]
        t_tokens = [self.sos] + self.tgt_tokenizer(t) + [self.eos]
        s_ids = self._numericalize(s_tokens, self.src_vocab)
        t_ids = self._numericalize(t_tokens, self.tgt_vocab)
        return torch.tensor(s_ids, dtype=torch.long), torch.tensor(t_ids, dtype=torch.long)
class Collator:
    # 可被 pickle 的 collate 函式封裝器（Windows multiprocessing safe）。
    # 建構時傳入 src/tgt 的 pad index，DataLoader 直接使用這個實例作 collate_fn。
    def __init__(self, pad_idx_src: int, pad_idx_tgt: int):
        self.pad_idx_src = pad_idx_src
        self.pad_idx_tgt = pad_idx_tgt

    def __call__(self, batch):
        #collate_fn 相同邏輯
        src_batch, tgt_batch = zip(*batch)
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=self.pad_idx_src)
        tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=self.pad_idx_tgt)
        return src_padded, tgt_padded


def collate_fn(batch, pad_idx_src, pad_idx_tgt):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx_src)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx_tgt)
    return src_padded, tgt_padded


def build_data_loaders(data_dir="data", batch_size=BZ, min_freq=MIN_F, n_w=0):
    #讀 train/valid/test
    train_pairs = r_parallel(data_dir, "train")
    valid_pairs = r_parallel(data_dir, "valid")
    test_pairs = r_parallel(data_dir, "test")

    #建詞表
    def src_tokens_gen():
        for s, _ in train_pairs:
            yield tokenize_en(s)

    def tgt_tokens_gen():
        for _, t in train_pairs:
            yield tokenize_zh(t)

    src_vocab = build_vocab(src_tokens_gen(), min_freq=min_freq)
    tgt_vocab = build_vocab(tgt_tokens_gen(), min_freq=min_freq)

    #設定 pad/unk/sos/eos index
    for sp in SPECIALS:
        if sp not in src_vocab["stoi"]:
            src_vocab["itos"].insert(0, sp)
            src_vocab["stoi"] = {tok: i for i, tok in enumerate(src_vocab["itos"]) }
        if sp not in tgt_vocab["stoi"]:
            tgt_vocab["itos"].insert(0, sp)
            tgt_vocab["stoi"] = {tok: i for i, tok in enumerate(tgt_vocab["itos"]) }

    pad_idx_src = src_vocab["stoi"]["<pad>"]
    pad_idx_tgt = tgt_vocab["stoi"]["<pad>"]

    #Datasets & Dataloaders
    train_dataset = ParallelDataset(train_pairs, src_vocab, tgt_vocab, src_tokenizer=tokenize_en, tgt_tokenizer=tokenize_zh)
    valid_dataset = ParallelDataset(valid_pairs, src_vocab, tgt_vocab, src_tokenizer=tokenize_en, tgt_tokenizer=tokenize_zh)
    test_dataset = ParallelDataset(test_pairs, src_vocab, tgt_vocab, src_tokenizer=tokenize_en, tgt_tokenizer=tokenize_zh)

    # 使用可 pickle 的 collator 實例（取代 lambda）
    collate_instance = Collator(pad_idx_src, pad_idx_tgt)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_instance, num_workers=n_w
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_instance, num_workers=n_w
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_instance, num_workers=n_w
    )

    return train_loader, valid_loader, test_loader, src_vocab, tgt_vocab

