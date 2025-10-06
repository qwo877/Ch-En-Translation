# model.py
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # shape (1, max_len, d_model)

    def forward(self, x):  # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.last_attn = None  #keep last attention for visualization

    def forward(self, query, key, value, mask=None):
        #query/key/value: (batch, seq_len, d_model)
        #mask: boolean mask where True indicates valid tokens, shape broadcastable to (batch, num_heads, q_len, k_len)
        #returns: (batch, seq_len, d_model)

        batch_size = query.size(0)
        q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)  # (batch, heads, q_len, d_k)
        k = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        v = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, heads, q_len, k_len)
        if mask is not None:
            # mask: True for valid tokens set invalid positions to -inf
            scores = scores.masked_fill(~mask, float("-1e9"))

        attn = torch.softmax(scores, dim=-1)  # (batch, heads, q_len, k_len)
        self.last_attn = attn.detach().cpu()
        out = torch.matmul(self.dropout(attn), v)  # (batch, heads, q_len, d_k)
        out = out.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.W_o(out)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        # x: (batch, seq_len, d_model), src_mask: (batch, 1, 1, src_len) boolean
        attn_out = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Masked self-attn
        attn1 = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn1))
        # Encoder-decoder cross-attention
        attn2 = self.cross_attn(x, enc_output, enc_output, mask=src_mask)  # src_mask used to mask encoder keys
        x = self.norm2(x + self.dropout(attn2))
        ff = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff))
        return x

def masks(src, tgt, src_pad_idx, tgt_pad_idx, device=None):
    #src: (batch, src_len)
    #tgt: (batch, tgt_len)
    #returns:
    # src_mask: shape (batch, 1, 1, src_len) boolean (True for non-pad)
    # tgt_mask: shape (batch, 1, tgt_len, tgt_len) boolean

    if device is None:
        device = src.device
    src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)  # (batch,1,1,src_len)
    tgt_pad_mask = (tgt != tgt_pad_idx).unsqueeze(1).unsqueeze(3)  # (batch,1,tgt_len,1)
    seq_len = tgt.size(1)
    no_peak = torch.triu(torch.ones((1, seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)  # (1, tgt_len, tgt_len)
    no_peak = no_peak.unsqueeze(1)  # (1,1,tgt_len,tgt_len)
    tgt_mask = tgt_pad_mask & ~no_peak  # broadcast -> (batch,1,tgt_len,tgt_len)
    return src_mask, tgt_mask

class TfMod(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_len=200, dropout=0.1):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        #src: (batch, src_len), tgt: (batch, tgt_len)
        #returns: logits (batch, tgt_len, tgt_vocab)
        # embeddings
        enc = self.src_emb(src) * math.sqrt(self.d_model)  # (batch, src_len, d_model)
        enc = self.pos_enc(enc)
        enc = self.dropout(enc)
        # encoder stack
        for layer in self.enc_layers:
            enc = layer(enc, src_mask)

        # decoder embeddings
        dec = self.tgt_emb(tgt) * math.sqrt(self.d_model)
        dec = self.pos_enc(dec)
        dec = self.dropout(dec)
        for layer in self.dec_layers:
            dec = layer(dec, enc, src_mask, tgt_mask)

        out = self.fc_out(dec)  # (batch, tgt_len, tgt_vocab)
        return out