#make_data_set
import os
import csv
import shutil
import tarfile
import bz2
import random
import requests
from tqdm import tqdm
import re
# ------------------ 參數 ------------------
OUT_DIR = "data"
SENTENCES_URL = "https://downloads.tatoeba.org/exports/sentences.csv"
LINKS_URLS = [
    "https://downloads.tatoeba.org/exports/links.csv",
    "https://downloads.tatoeba.org/exports/links.tar.bz2",
    "https://downloads.tatoeba.org/exports/links.csv.bz2"
]
# 語言代碼（Tatoeba 用 ISO639-3）— 英文是 "eng"，中文常見為 "cmn" (Mandarin)，也可能看到 "zho"
EN_CODES = {"eng"}
ZH_CODES = {"cmn", "zho", "chi", "cmn-Hant", "cmn-Hans"}
MAX_PAIRS = 120_000     # 取出最多多少對
MIN_CHAR_LEN = 5        # 最短字元數
MAX_CHAR_LEN = 300      # 最長字元數
TRAIN_RATIO = 0.90
VALID_RATIO = 0.05
TEST_RATIO = 0.05
RANDOM_SEED = 42
REMOVE_UNK = True
# ----------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
TMP_DIR = "tmp_tatoeba"
os.makedirs(TMP_DIR, exist_ok=True)

def normal(s: str):
    return not bool(re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", s))

def dow_s(url, out_path, desc=None):
    print(f"下載：{url}")
    r = requests.get(url, stream=True, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"下載失敗（HTTP {r.status_code}）: {url}")
    total = int(r.headers.get("content-length", 0))
    with open(out_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=desc or os.path.basename(out_path)) as pbar:
        for chunk in r.iter_content(chunk_size=1024*32):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    return out_path


def extract(path):
    # 如果是 .bz2、.tar.bz2、.tar 等，嘗試解壓並回傳解出的 links 檔完整路徑
    lp = path.lower()
    if lp.endswith(".tar.bz2") or lp.endswith(".tar.bz2"):
        print("解壓 tar.bz2 ...")
        with tarfile.open(path, "r:bz2") as tar:
            tar.extractall(path=TMP_DIR)
        # 在 tmp 目錄找 links.csv
        for root, dirs, files in os.walk(TMP_DIR):
            for f in files:
                if f.lower().startswith("links") and f.lower().endswith(".csv"):
                    return os.path.join(root, f)
    elif lp.endswith(".bz2") and not path.endswith(".tar.bz2"):
        # 單純的 bz2 (直接解壓出 csv)
        print("解壓 .bz2 ...")
        out = path[:-4]
        with bz2.open(path, "rb") as fr, open(out, "wb") as fw:
            shutil.copyfileobj(fr, fw)
        return out
    else:
        # 不是壓縮檔，直接回傳
        return path

def load_s(sentences_csv_path, keep_langs):
    print("解析 sentences.csv")
    mapping = {}
    removed_cnt = 0
    total = 0
    with open(sentences_csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in tqdm(reader, desc="讀 sentences", unit="line"):
            total += 1
            if len(row) < 3:
                continue
            sid = row[0].strip()
            lang = row[1].strip()
            text = row[2].strip()
            if not sid or not lang:
                continue
            if lang in keep_langs:
                # 濾掉明顯有 <unk> 的句子（大寫/小寫都算）
                if REMOVE_UNK and ("<unk>" in text or "<UNK>" in text):
                    removed_cnt += 1
                    continue
                if not normal(text):
                    removed_cnt += 1
                    continue
                try:
                    mapping[int(sid)] = (lang, text)
                except:
                    continue
    print(f"保留句子數（指定語言）: {len(mapping)}；總讀入行數: {total}；移除: {removed_cnt}")
    return mapping

def load_l(links_csv_path, sentences_map, max_pairs=None):
    print("解析 links.csv 並抽取英中平行句對")
    pairs = []
    seen = set()
    removed_pair_counts = {"missing":0, "bad_len":0, "contains_unk":0}
    with open(links_csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in tqdm(reader, desc="讀 links", unit="line"):
            if len(row) < 2:
                continue
            try:
                a = int(row[0].strip()); b = int(row[1].strip())
            except:
                continue
            if a not in sentences_map or b not in sentences_map:
                removed_pair_counts["missing"] += 1
                continue
            la, ta = sentences_map[a]
            lb, tb = sentences_map[b]
            # 檢查是否為英與中文的配對
            if (la in EN_CODES and lb in ZH_CODES) or (lb in EN_CODES and la in ZH_CODES):
                if la in EN_CODES:
                    en, zh = ta, tb
                else:
                    en, zh = tb, ta
                en = en.strip()
                zh = zh.strip()
                # 過濾控制字元或太長/太短
                if len(en) < MIN_CHAR_LEN or len(zh) < MIN_CHAR_LEN:
                    removed_pair_counts["bad_len"] += 1
                    continue
                if len(en) > MAX_CHAR_LEN or len(zh) > MAX_CHAR_LEN:
                    removed_pair_counts["bad_len"] += 1
                    continue
                # 如果任一邊包含 "<unk>" 則完全跳過（使用者要求）
                if REMOVE_UNK and ("<unk>" in en or "<UNK>" in en or "<unk>" in zh or "<UNK>" in zh):
                    removed_pair_counts["contains_unk"] += 1
                    continue
                # 再檢查是否含控制字元
                if not normal(en) or not normal(zh):
                    removed_pair_counts["bad_len"] += 1
                    continue
                # 去重與去掉完全相同內容
                key = (en, zh)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((en, zh))
                if max_pairs and len(pairs) >= max_pairs:
                    break
    print(f"抽出平行句對數: {len(pairs)}；被移除的 pairs: {removed_pair_counts}")
    return pairs

def split_write(pairs, out_dir, train_ratio=0.90, valid_ratio=0.05, test_ratio=0.05):
    os.makedirs(out_dir, exist_ok=True)
    print("shuffle 並切分資料集...")
    random.seed(RANDOM_SEED)
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    train = pairs[:n_train]
    valid = pairs[n_train:n_train+n_valid]
    test = pairs[n_train+n_valid:]
    print(f"train={len(train)}, valid={len(valid)}, test={len(test)}")
    def write_split(lst, prefix):
        with open(os.path.join(out_dir, f"{prefix}.en"), "w", encoding="utf-8", newline="\n") as fe, \
             open(os.path.join(out_dir, f"{prefix}.zh"), "w", encoding="utf-8", newline="\n") as fz:
            for en, zh in lst:
                fe.write(en.replace("\n"," ") + "\n")
                fz.write(zh.replace("\n"," ") + "\n")
    write_split(train, "train")
    write_split(valid, "valid")
    write_split(test, "test")
    print(f"已輸出到 {out_dir}/")

def main():
    try:
        # 下載 sentences.csv
        sentences_local = os.path.join(TMP_DIR, "sentences.csv")
        if not os.path.exists(sentences_local):
            dow_s(SENTENCES_URL, sentences_local, desc="sentences.csv")
        else:
            print("已存在 sentences.csv，略過下載。")

        # 下載 links
        links_local = None
        for url in LINKS_URLS:
            try:
                fname = os.path.join(TMP_DIR, os.path.basename(url))
                if os.path.exists(fname):
                    links_local = fname
                    print(f"已存在 {fname}，略過下載。")
                    break
                dow_s(url, fname, desc=os.path.basename(url))
                links_local = fname
                break
            except Exception as e:
                print(f"下載失敗：{url}，錯誤：{e}")
                continue
        if links_local is None:
            raise RuntimeError("無法下載 links 檔案，檢查網址或網路。")

        # 解壓
        links_csv = extract(links_local)

        # 解析 sentences
        keep_langs = set()
        keep_langs.update(EN_CODES)
        keep_langs.update(ZH_CODES)
        sentences_map = load_s(sentences_local, keep_langs)

        # 解析 links 並抽對 (最多 MAX_PAIRS)
        pairs = load_l(links_csv, sentences_map, max_pairs=MAX_PAIRS)

        if len(pairs) == 0:
            raise RuntimeError("找不到任何平行句對。")

        # 切分並寫檔
        split_write(pairs, OUT_DIR, TRAIN_RATIO, VALID_RATIO, TEST_RATIO)

        print("OK")
    finally:
        # 清理 tmp
        print("清理暫存檔...")
        try:
            shutil.rmtree(TMP_DIR)
        except Exception:
            pass

if __name__ == "__main__":
    main()
