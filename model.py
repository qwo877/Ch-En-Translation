# model.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, 
                 d_model=256, n_heads=8, n_layers=3, d_ff=512, 
                 src_pad_idx=1, tgt_pad_idx=1, dropout=0.1, max_len=500):
        super().__init__()
        
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.d_model = d_model
        
        self.encoder_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_idx)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, n_layers)
        
        self.decoder_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_idx)
        dec_layer = nn.TransformerDecoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, n_layers)
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        
        return (src == self.src_pad_idx)

    def make_tgt_mask(self, tgt):
        
        tgt_len = tgt.shape[1]
        tgt_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=tgt.device), diagonal=1).bool()
        return tgt_mask

    def forward(self, src, tgt):

        src_key_padding_mask = self.make_src_mask(src)
        tgt_key_padding_mask = self.make_src_mask(tgt)
        tgt_mask = self.make_tgt_mask(tgt)
        
        src_emb = self.pos_enc(self.encoder_emb(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.decoder_emb(tgt) * math.sqrt(self.d_model))
        
        enc_output = self.transformer_encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        
        output = self.transformer_decoder(
            tgt_emb, enc_output, 
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        return self.fc_out(output)