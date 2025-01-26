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


class CrossAttention(nn.Module):
    def __init__(self, input_dim_q, input_dim_kv, hidden_size=32, num_heads=4):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = np.sqrt(self.head_dim)

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(input_dim_q, hidden_size)  # Query projection
        self.k_proj = nn.Linear(input_dim_kv, hidden_size)  # Key projection
        self.v_proj = nn.Linear(input_dim_kv, hidden_size)  # Value projection
        self.out_proj = nn.Linear(hidden_size, input_dim_q)  # Output projection

    def forward(self, query, key_value):
        """
        Args:
            query: Tensor of shape [batch_size, seq_len_q, input_dim_q]
            key_value: Tensor of shape [batch_size, seq_len_kv, input_dim_kv]
        Returns:
            output: Tensor of shape [batch_size, seq_len_q, input_dim_q]
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_kv = key_value.shape[1]

        # Linear projections
        q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        k = self.k_proj(key_value).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(key_value).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention scores: [batch_size, num_heads, seq_len_q, seq_len_kv]
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Attention weights: [batch_size, num_heads, seq_len_q, seq_len_kv]
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Attention output: [batch_size, num_heads, seq_len_q, head_dim]
        attention_output = torch.matmul(attention_weights, v)

        # Combine heads and project back to input_dim_q
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len_q, -1)
        output = self.out_proj(attention_output)

        return output
    
    
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



class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1,device='cuda'):
        super(TransformerEncoderLayer, self).__init__()
        
        # Multi-Head Attention
        self.self_attention = MultiHeadAttention(embed_dim, num_heads)
        self.device = device
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
        linear_layer = nn.Linear(x.shape[0], 1).to(self.device)  # 从 1000 维变到 1 维
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈神经网络：残差连接 + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        p1_transformed = linear_layer(x.transpose(1, 2)).squeeze(-1)  # 输出形状变为 (1000, 8)
        print(p1_transformed.shape)
        return p1_transformed


class TransformerEncoderLayer_decay(nn.Module):
    def __init__(self, embed_dim,output_dim,num_heads, ff_dim, dropout=0.1,device='cuda'):
        super(TransformerEncoderLayer_decay, self).__init__()
        self.device = device
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
        linear_layer = nn.Linear(x.shape[0], 1).to(self.device)  # 从 1000 维变到 1 维
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈神经网络：残差连接 + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(self.dropout(ffn_output))
        p1_transformed = linear_layer(x.transpose(1, 2)).squeeze(-1)  # 输出形状变为 (1000, 8)
        return p1_transformed
    


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout=0.1,ifdecay=False):
        super(TransformerEncoder, self).__init__()
        if ifdecay:
            self.embedddecay_list = [embed_dim - i * (embed_dim // num_layers) for i in range(num_layers)]
        else:
            self.embedddecay_list = [embed_dim for i in range(num_layers)]
        if self.embedddecay_list[-1] <=0 :
            raise ValueError("The last layer's embed_dim is less than 0")
        if embed_dim==1 and ifdecay:
            raise ValueError("The embed_dim is 1, cannot decay")
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





class TransformerDecoderLayercat(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerDecoderLayercat, self).__init__()
        
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

        enc_output = enc_output.squeeze(1)
        attn_output = torch.concat([x, enc_output], dim=1)
        attn_output = nn.Linear(attn_output.shape[1],1)(attn_output)  
        
        return x




class TransformerDecoderLayer_pre(nn.Module):
    def __init__(self, embed_dim, enc_num, num_heads, ff_dim, dropout=0.1):
        super(TransformerDecoderLayer_pre, self).__init__()
        self.embed_dim = embed_dim
        self.self_attention = MultiHeadAttention(embed_dim, num_heads)
        self.encoder_decoder_attention = CrossAttention(embed_dim, enc_num)
        self.positional_encoding = nn.Parameter(torch.randn(1, 2000, embed_dim))  # 动态位置编码
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, self_mask=None, enc_mask=None):
        # 加入位置编码
        seq_len = x.size(0)
        x = x.unsqueeze(0)
        enc_output = enc_output.unsqueeze(1)
        x = x + self.positional_encoding[:, :seq_len, :]
        x = x.permute(1, 0, 2)
        _x = x

        x = self.self_attention(query=x, key=x, value=x, mask=self_mask)
        x = self.dropout(x)
        x = self.norm1(x + _x)
        _x = x
        x = self.encoder_decoder_attention(query=x, key_value=enc_output)
        x = self.dropout(x)
        x = self.norm2(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm3(x + _x)

        x = x.view(x.shape[0], -1)
        linear_layer = nn.Linear(x.shape[1], self.embed_dim).cuda()  # 或 .to('cuda')
        x = linear_layer(x)
        return x




class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim,enc_num, num_heads, ff_dim, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.embed_dim = embed_dim
        # 多头自注意力
        self.self_attention = MultiHeadAttention(embed_dim, num_heads)
        
        # Encoder-Decoder Attention
        self.encoder_decoder_attention = CrossAttention(embed_dim,enc_num)
        
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
        x = x.unsqueeze(1)
        enc_output = enc_output.unsqueeze(1)
    
        _x = x
        x = self.self_attention(query=x, key=x, value=x, mask=self_mask)

        x = self.dropout(x)
        x = self.norm1(x + _x)
        _x = x

    
        x = self.encoder_decoder_attention(query=x, key_value=enc_output)
        x = self.dropout(x)
        x = self.norm2(x + _x)

        # Feed-Forward Network
        _x = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm3(x + _x)

        x = x.view(x.shape[0], -1)
        linear_layer = nn.Linear(x.shape[1], self.embed_dim).cuda()  # 或 .to('cuda')
        x = linear_layer(x)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, decoder_embed_dim,embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        
        # 创建多个 Transformer Decoder 层
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(decoder_embed_dim, embed_dim,num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, enc_output, self_mask=None, enc_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, self_mask, enc_mask)
        return x



class TransformerBlock(nn.Module):
    def __init__(self, num_layers, embed_dim,decoder_embed_dim, num_heads, ff_dim, dropout=0.1,ifdecay=False):
        super(TransformerBlock, self).__init__()
        
        # Encoder 和 Decoder
        self.encoder = TransformerEncoder(num_layers, embed_dim, num_heads, ff_dim, dropout,ifdecay)
        self.decoder = TransformerDecoder(num_layers, decoder_embed_dim,self.encoder.embedddecay_list[-1], num_heads, ff_dim, dropout)
        self.embedding_layer = nn.Embedding(101, embed_dim)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        enc_output = self.encoder(src, src_mask)
        
        # 解码器部分
        dec_output = self.decoder(tgt, enc_output, tgt_mask, memory_mask)
        
        return dec_output


