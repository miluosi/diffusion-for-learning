import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch.optim as optim


import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 每个头的维度
        
        # 可训练的投影矩阵
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性投影：计算 Q, K, V
        Q = self.q_proj(query)  # [batch_size, seq_len, embed_dim]
        K = self.k_proj(key)    # [batch_size, seq_len, embed_dim]
        V = self.v_proj(value)  # [batch_size, seq_len, embed_dim]
        
        # 分头：变形为 [batch_size, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数：scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        
        # 注意力加权
        attended = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len, head_dim]
        
        # 拼接多头的结果
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        # 输出投影
        output = self.out_proj(attended)  # [batch_size, seq_len, embed_dim]
        return output




import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        # Multi-Head Attention
        self.self_attention = MultiHeadAttention(embed_dim, num_heads)
        
        # 前馈神经网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):

        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈神经网络：残差连接 + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        print(x.shape)
        return x


class TransformerEncoderLayer_decay(nn.Module):
    def __init__(self, embed_dim,output_dim,num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderLayer_decay, self).__init__()
        
        # Multi-Head Attention
        self.self_attention = MultiHeadAttention(embed_dim, num_heads)
        
        # 前馈神经网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, output_dim)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):

        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈神经网络：残差连接 + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(self.dropout(ffn_output))
        print(x.shape)
        return x
    


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout=0.1,ifdecay=False):
        super(TransformerEncoder, self).__init__()
        self.embedddecay_list = [embed_dim - i * (embed_dim // num_layers) for i in range(num_layers)]


        self.embedd_decay = nn.ModuleList([
            TransformerEncoderLayer_decay(self.embedddecay_list[i],self.embedddecay_list[i+1], num_heads, ff_dim, dropout)
            for i in range(num_layers-1)])
        
        self.ifdecay = ifdecay
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        if self.ifdecay:
            for layers in self.embedd_decay:
                x = layers(x,mask)
        else:
            for layer in self.layers:
                x = layer(x, mask)
        return x





class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        
        # 多头自注意力
        self.self_attention = MultiHeadAttention(embed_dim, num_heads)
        
        # Encoder-Decoder Attention
        self.encoder_decoder_attention = MultiHeadAttention(embed_dim, num_heads)
        
        # 前馈神经网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, self_mask=None, enc_mask=None):
        print(enc_output.shape)
        enc_output = enc_output.squeeze(1)
        attn_output = torch.concat([x, enc_output], dim=1)
        attn_output = nn.Linear(attn_output.shape[1],1)(attn_output)  
        
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        
        # 创建多个 Transformer Decoder 层
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, enc_output, self_mask=None, enc_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, self_mask, enc_mask)
        return x



class TransformerBlock(nn.Module):
    def __init__(self, num_layers, embed_dim,decoder_embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        # Encoder 和 Decoder
        self.encoder = TransformerEncoder(num_layers, embed_dim, num_heads, ff_dim, dropout)
        self.decoder = TransformerDecoder(num_layers, decoder_embed_dim, num_heads, ff_dim, dropout)
        self.embedding_layer = nn.Embedding(101, embed_dim)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        enc_output = self.encoder(src, src_mask)
        
        # 解码器部分
        dec_output = self.decoder(tgt, enc_output, tgt_mask, memory_mask)
        
        return dec_output


