# preprocess.py
import os
import random
import requests
import tarfile
import csv
import re
from tqdm import tqdm
from config import Config
from opencc import OpenCC

# --- 設定 ---
URL_SENTENCES = "https://downloads.tatoeba.org/exports/sentences.tar.bz2"
URL_LINKS = "https://downloads.tatoeba.org/exports/links.tar.bz2"

TMP_DIR = "tmp_data"

cc = OpenCC('s2t')

RAW_MAX = 400000

def download_file(url, dest):
    if os.path.exists(dest):
        print(f"{dest} 已存在，跳過下載")
        return
    print(f"下載 {url} ...")
    try:
        r = requests.get(url, stream=True)
        total_size = int(r.headers.get('content-length', 0))
        with open(dest, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc=dest) as pbar:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk: 
                    f.write(chunk)
                    pbar.update(len(chunk))
    except Exception as e:
        print(f"下載失敗: {e}")
        if os.path.exists(dest): os.remove(dest)
        raise

def extract_tar_bz2(archive_path, target_filename):
    print(f"正在解壓縮 {archive_path} ...")
    
    target_path = os.path.join(TMP_DIR, target_filename)
    
    if os.path.exists(target_path):
        print(f" -> {target_path} 已存在，跳過解壓")
        return target_path

    with tarfile.open(archive_path, "r:bz2") as tar:
        tar.extractall(TMP_DIR)
    
    if not os.path.exists(target_path):
        for root, dirs, files in os.walk(TMP_DIR):
            if target_filename in files:
                return os.path.join(root, target_filename)
        raise FileNotFoundError(f"解壓後找不到 {target_filename}")
        
    return target_path

def clean_str(s):
    return re.sub(r"[\x00-\x1f]", "", s).strip()

def main():
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    s_archive = os.path.join(TMP_DIR, "sentences.tar.bz2")
    l_archive = os.path.join(TMP_DIR, "links.tar.bz2")
    
    download_file(URL_SENTENCES, s_archive)
    download_file(URL_LINKS, l_archive)
    
    s_csv = extract_tar_bz2(s_archive, "sentences.csv")
    l_csv = extract_tar_bz2(l_archive, "links.csv")
    
    print("-" * 30)
    print(f"Sentences CSV: {s_csv}")
    print(f"Links CSV: {l_csv}")
    print("-" * 30)

    print("讀取句子中")
    s_map = {}
    
    with open(s_csv, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in tqdm(reader, desc="Scanning Sentences"):
            if len(row) < 3: continue
            sid, lang, text = row[0], row[1], row[2]
            
            if lang in ['eng', 'cmn', 'zho']:
                s_map[sid] = (lang, clean_str(text))
    
    print(f"有效句子數量: {len(s_map)}")
    if len(s_map) == 0:
        raise ValueError("沒有讀取到任何有效句子")

    print("配對並建立平行語料")
    pairs = []
    seen = set()
    
    with open(l_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in tqdm(reader, desc="Scanning Links"):
            if len(row) < 2: continue
            id1, id2 = row[0], row[1]
            
            if id1 not in s_map or id2 not in s_map: continue
            
            l1, t1 = s_map[id1]
            l2, t2 = s_map[id2]
            
            en, zh = None, None
            
            if l1 == 'eng' and l2 in ['cmn', 'zho']:
                en, zh = t1, t2
            elif l2 == 'eng' and l1 in ['cmn', 'zho']:
                en, zh = t2, t1
            
            if en and zh:
                if 2 < len(en) < 200 and 1 < len(zh) < 200:
                    pair_key = (en, zh)
                    if pair_key not in seen:
                        zh_trad = cc.convert(zh)
                        pairs.append((en, zh_trad))
                        seen.add(pair_key)
                        
                if len(pairs) >= RAW_MAX: 
                    break
    
    print(f"配對句組: {len(pairs)}")
    
    if len(pairs) == 0:
        print("沒有找到任何配對")
        return

    print("寫入檔案")
    random.shuffle(pairs)
    split_idx = int(len(pairs) * 0.9)
    train_data = pairs[:split_idx]
    valid_data = pairs[split_idx:]
    
    def write_file(name, data):
        path_en = os.path.join(Config.DATA_DIR, f"{name}.en")
        path_zh = os.path.join(Config.DATA_DIR, f"{name}.zh")
        
        with open(path_en, 'w', encoding='utf-8') as fe, \
             open(path_zh, 'w', encoding='utf-8') as fz:
            for en, zh in data:
                fe.write(en + "\n")
                fz.write(zh + "\n")
        print(f"已寫入 {path_en} ({len(data)} 句)")

    write_file("train", train_data)
    write_file("valid", valid_data)
    print("資料處理完成")

if __name__ == "__main__":
    main()

