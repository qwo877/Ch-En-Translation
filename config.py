# config.py
import torch

class Config:
    # --- 路徑與存檔 ---
    DATA_DIR = "data"
    SAVE_PATH = "transformer_model.pt"
    
    # --- Tokenizer 設定 (使用 HuggingFace) ---
    TOKENIZER_NAME = "bert-base-multilingual-cased"
    
    # --- 資料處理 ---
    MAX_LEN = 64        # 句子最大長度
    BATCH_SIZE = 32     # 批次大小
    
    # --- 模型參數  ---
    d_model = 512       # 嵌入向量維度
    n_layers = 4        # 編碼器/解碼器層數
    n_heads = 8         # 注意力頭數
    d_ff = 2048         # 前饋網路維度
    dropout = 0.1
    
    # --- 訓練參數 ---
    EPOCHS = 20         #輪數
    LR = 0.0001         #學習率
    LABEL_SMOOTH = 0.1
    GRAD_CLIP = 1.0
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 特殊 Token
    UNK_IDX = 0
    PAD_IDX = 1
    SOS_IDX = 2
    EOS_IDX = 3
    SPECIALS = ["<unk>", "<pad>", "<sos>", "<eos>"]