# predict.py
import torch
from transformers import AutoTokenizer
from config import Config
from model import Transformer
import re
from opencc import OpenCC

cc = OpenCC('s2t')
tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)

def post_process(text):
    text = cc.convert(text)
    text = re.sub(r'(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])', '', text)
    text = re.sub(r'\s+([。，！？、])', r'\1', text)
    
    return text

def load_model(path):
    checkpoint = torch.load(path, map_location=Config.DEVICE, weights_only=False)
    cfg = checkpoint['config']
    
    model = Transformer(
        src_vocab_size=cfg['vocab_size'], 
        tgt_vocab_size=cfg['vocab_size'],
        d_model=Config.d_model,
        n_heads=Config.n_heads, 
        n_layers=Config.n_layers, 
        d_ff=Config.d_ff,
        src_pad_idx=cfg['pad_idx'], 
        tgt_pad_idx=cfg['pad_idx'],
        dropout=Config.dropout
    ).to(Config.DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def translate(text, model, beam_size=3, max_len=60):
    model.eval()
    
    src_enc = tokenizer(text, return_tensors='pt').to(Config.DEVICE)
    src = src_enc['input_ids']
    
    decoder_input = torch.tensor([[tokenizer.cls_token_id]], device=Config.DEVICE)
    
    candidates = [(0.0, decoder_input)]
    
    completed_sequences = []
    
    for _ in range(max_len):
        new_candidates = []
        
        for score, seq in candidates:
            if seq.size(1) > max_len: continue
            
            if seq[0, -1].item() == tokenizer.sep_token_id:
                completed_sequences.append((score, seq))
                continue
                
            with torch.no_grad():
                output = model(src, seq) 
                next_token_logits = output[:, -1, :]
                probs = torch.log_softmax(next_token_logits, dim=-1)
            
            top_probs, top_ids = probs.topk(beam_size)
            
            for i in range(beam_size):
                new_score = score + top_probs[0, i].item()
                next_token = top_ids[0, i].unsqueeze(0).unsqueeze(0)
                new_seq = torch.cat([seq, next_token], dim=1)
                new_candidates.append((new_score, new_seq))
        
        candidates = sorted(new_candidates, key=lambda x: x[0], reverse=True)[:beam_size]
        
        if len(completed_sequences) >= beam_size:
            break
            
    if not completed_sequences:
        completed_sequences = candidates

    best_score, best_seq = max(completed_sequences, key=lambda x: x[0])
    
    decoded = tokenizer.decode(best_seq.squeeze(), skip_special_tokens=True)
    decoded = post_process(decoded)

    return decoded

if __name__ == "__main__":
    model = load_model(Config.SAVE_PATH)
    
    sentences = [
        "The meeting starts at nine sharp.",
        "I left my umbrella in the office yesterday.",
        "She only said she would try.",
        "I don’t think he is wrong.",
        "When Tom met Jack, he was nervous.",
        "If I had known earlier, I would have helped.",
        "The decision was made without consulting the team.",
        "He is as stubborn as he is talented.",
        "The file is 1.5 GB, not 1,5 GB."
    ]
    
    print("-" * 30)
    for s in sentences:
        print(f"原文: {s}")
        print(f"翻譯: {translate(s, model)}")
        print("-" * 30)
    s=input("輸入英文:")
    print(f"原文: {s}")
    print(f"翻譯: {translate(s, model)}")