import os
import csv
import shutil
import tarfile
import bz2
import random
import requests
from tqdm import tqdm

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
MAX_PAIRS = 25_000     # 取出最多多少對
MIN_CHAR_LEN = 1        # 最短字元數
MAX_CHAR_LEN = 300      # 最長字元數
TRAIN_RATIO = 0.90
VALID_RATIO = 0.05
TEST_RATIO = 0.05
RANDOM_SEED = 42
# ----------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
TMP_DIR = "tmp_tatoeba"
os.makedirs(TMP_DIR, exist_ok=True)

def download_stream(url, out_path, desc=None):
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

def try_download_links():
    # 嘗試 links.csv、links.csv.bz2、links.tar.bz2 等
    for url in LINKS_URLS:
        try:
            outfn = os.path.join(TMP_DIR, os.path.basename(url))
            download_stream(url, outfn)
            return outfn
        except Exception as e:
            print(f"嘗試 {url} 失敗：{e}")
    raise RuntimeError("無法下載 links 檔案，請檢查網路或網址。")

def extract_if_bz2_or_tar(path):
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

def load_sentences(sentences_csv_path, keep_langs):
    print("解析 sentences.csv（只保留指定語言以節省記憶體）...")
    mapping = {}
    # sentences.csv 每行: id \t lang \t text
    with open(sentences_csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in tqdm(reader, desc="讀 sentences", unit="line"):
            if len(row) < 3:
                continue
            sid = row[0].strip()
            lang = row[1].strip()
            text = row[2].strip()
            if not sid or not lang:
                continue
            if lang in keep_langs:
                try:
                    mapping[int(sid)] = (lang, text)
                except:
                    continue
    print(f"保留句子數（指定語言）: {len(mapping)}")
    return mapping

def load_links(links_csv_path, sentences_map, max_pairs=None):
    print("解析 links.csv 並抽取英中平行句對...")
    pairs = []
    seen = set()
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
                continue
            la, ta = sentences_map[a]
            lb, tb = sentences_map[b]
            # 檢查是否為英與中文的配對
            if (la in EN_CODES and lb in ZH_CODES) or (lb in EN_CODES and la in ZH_CODES):
                if la in EN_CODES:
                    en, zh = ta, tb
                else:
                    en, zh = tb, ta
                # 基本清理
                en = en.strip()
                zh = zh.strip()
                if len(en) < MIN_CHAR_LEN or len(zh) < MIN_CHAR_LEN:
                    continue
                if len(en) > MAX_CHAR_LEN or len(zh) > MAX_CHAR_LEN:
                    continue
                # 去重與去掉完全相同內容
                key = (en, zh)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((en, zh))
                if max_pairs and len(pairs) >= max_pairs:
                    break
    print(f"抽出平行句對數: {len(pairs)}")
    return pairs

def split_and_write(pairs, out_dir, train_ratio=0.90, valid_ratio=0.05, test_ratio=0.05):
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
        fe = open(os.path.join(out_dir, f"{prefix}.en"), "w", encoding="utf-8")
        fz = open(os.path.join(out_dir, f"{prefix}.zh"), "w", encoding="utf-8")
        for en, zh in lst:
            fe.write(en.replace("\n"," ") + "\n")
            fz.write(zh.replace("\n"," ") + "\n")
        fe.close(); fz.close()
    write_split(train, "train")
    write_split(valid, "valid")
    write_split(test, "test")
    print(f"已輸出到 {out_dir}/")

def main():
    try:
        # 下載 sentences.csv
        sentences_local = os.path.join(TMP_DIR, "sentences.csv")
        if not os.path.exists(sentences_local):
            download_stream(SENTENCES_URL, sentences_local, desc="sentences.csv")
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
                download_stream(url, fname, desc=os.path.basename(url))
                links_local = fname
                break
            except Exception as e:
                print(f"下載失敗：{url}，錯誤：{e}")
                continue
        if links_local is None:
            raise RuntimeError("無法下載 links 檔案，請手動檢查網址或網路。")

        # 解壓
        links_csv = extract_if_bz2_or_tar(links_local)

        # 解析 sentences
        keep_langs = set()
        keep_langs.update(EN_CODES)
        keep_langs.update(ZH_CODES)
        sentences_map = load_sentences(sentences_local, keep_langs)

        # 解析 links 並抽對 (最多 MAX_PAIRS)
        pairs = load_links(links_csv, sentences_map, max_pairs=MAX_PAIRS)

        if len(pairs) == 0:
            raise RuntimeError("找不到任何平行句對。請檢查 sentences/links 是否為最新版本、或語言代碼是否正確。")

        # 切分並寫檔
        split_and_write(pairs, OUT_DIR, TRAIN_RATIO, VALID_RATIO, TEST_RATIO)

        print("完成！請用你的 read_parallel / build_data_loaders 直接讀取 data/ 目錄中的檔案。")
    finally:
        # 清理 tmp
        print("清理暫存檔...")
        try:
            shutil.rmtree(TMP_DIR)
        except Exception:
            pass

if __name__ == "__main__":
    main()
