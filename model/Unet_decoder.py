import numpy as np
import random
import os
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class UNet1D(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dims=(64, 128, 256)):
        super(UNet1D, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.condition_transform = nn.Linear(condition_dim, hidden_dims[-1])
        self.input_dim = input_dim
        # Encoder layers
        in_channels = 1
        for h_dim in hidden_dims:
            self.encoders.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, h_dim, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2, padding=1)  # Adjust padding
                )
            )
            in_channels = h_dim

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Decoder layers
        for i, h_dim in enumerate(reversed(hidden_dims)):

            self.decoders.append(
                nn.Sequential(
                    nn.ConvTranspose1d(h_dim*2, h_dim//2, kernel_size=2, stride=2),  # Match channels here
                    nn.ReLU()
                )
            )

        # Final layer should produce output with 1 channel
        self.final_layer = nn.Conv1d(hidden_dims[0]//2, 1, kernel_size=1)

    def forward(self, x, condition):
        # Reshape input to [batch_size, 1, input_dim]
        x = x.unsqueeze(1)

        # Encoder
        encodings = []
        for encoder in self.encoders:
            x = encoder(x)
            encodings.append(x)

        # Bottleneck
        condition_vector = self.condition_transform(condition).unsqueeze(-1)
        x = self.bottleneck(x + condition_vector)

        # Decoder with skip connections
        for i, decoder in enumerate(self.decoders):
            # Skip connection: get the corresponding encoding
            skip_encoding = encodings[-(i+1)]  # Get encoding from corresponding encoder layer
            
            # Ensure skip encoding is the same size as the current x
            if skip_encoding.size(2) != x.size(2):
                skip_encoding = F.interpolate(skip_encoding, size=x.size(2), mode='linear', align_corners=False)

            # Concatenate skip connection with decoder output
            x = torch.cat([x, skip_encoding], dim=1)
            
            x = decoder(x)

        # Final output: Ensure output has a single channel
        x = self.final_layer(x)
        linear_layer = nn.Linear(x.shape[2], self.input_dim).cuda()
        x = linear_layer(x.squeeze(1)) 
        return x

    


class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_size=32, num_heads=4):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = np.sqrt(self.head_dim)

        self.q_proj = nn.Linear(input_dim, hidden_size)
        self.k_proj = nn.Linear(input_dim, hidden_size)
        self.v_proj = nn.Linear(input_dim, hidden_size)
        self.out_proj = nn.Linear(hidden_size, input_dim)

    def forward(self, x):
        # x: [batch_size, channels, seq_len]
        batch_size, channels, seq_len = x.shape

        # Reshape for multi-head attention
        q = self.q_proj(x).view(batch_size, channels, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(batch_size, channels, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(batch_size, channels, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Attention output
        attention_output = torch.matmul(attention_weights, v)
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, channels, -1)

        # Project back to input dimension
        return self.out_proj(attention_output)





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





class UNet1DWithAttention(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dims=(64, 128, 256)):
        super(UNet1DWithAttention, self).__init__()
        self.encoders = nn.ModuleList()
        self.attention_layers = nn.ModuleList()  # Attention layers in the encoder
        self.decoders = nn.ModuleList()
        self.condition_transform = nn.Linear(condition_dim, hidden_dims[-1])
        self.input_dim = input_dim
        
        # Encoder layers
        in_channels = 1
        for h_dim in hidden_dims:
            self.encoders.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, h_dim, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2, padding=1)  # Adjust padding
                )
            )
            self.attention_layers.append(SelfAttention(h_dim))  # Add self-attention layer
            in_channels = h_dim

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.bottleneck_attention = SelfAttention(hidden_dims[-1])  # Attention in bottleneck

        # Decoder layers
        for i, h_dim in enumerate(reversed(hidden_dims)):
            self.decoders.append(
                nn.Sequential(
                    nn.ConvTranspose1d(h_dim*2, h_dim//2, kernel_size=2, stride=2),  # Match channels here
                    nn.ReLU()
                )
            )

        # Final layer should produce output with 1 channel
        self.final_layer = nn.Conv1d(hidden_dims[0]//2, 1, kernel_size=1)

    def forward(self, x, condition):
        # Reshape input to [batch_size, 1, input_dim]
        x = x.unsqueeze(1)
        # Encoder
        encodings = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            x = x.permute(0, 2, 1) 
            x = self.attention_layers[i](x)  # Apply self-attention
            x = x.permute(0, 2, 1)
            encodings.append(x)

        # Bottleneck
        condition_vector = self.condition_transform(condition).unsqueeze(-1)
        x = self.bottleneck(x + condition_vector)
        x = x.permute(0, 2, 1) 
        x = self.bottleneck_attention(x)  # Apply self-attention in bottleneck
        x = x.permute(0, 2, 1) 
        # Decoder with skip connections
        for i, decoder in enumerate(self.decoders):
            # Skip connection: get the corresponding encoding
            skip_encoding = encodings[-(i+1)]  # Get encoding from corresponding encoder layer
            
            # Ensure skip encoding is the same size as the current x
            if skip_encoding.size(2) != x.size(2):
                skip_encoding = F.interpolate(skip_encoding, size=x.size(2), mode='linear', align_corners=False)

            # Concatenate skip connection with decoder output
            x = torch.cat([x, skip_encoding], dim=1)
            x = decoder(x)

        # Final output: Ensure output has a single channel
        x = self.final_layer(x)
        linear_layer = nn.Linear(x.shape[2], self.input_dim).cuda()
        x = linear_layer(x.squeeze(1)) 
        return x




class UNet1DWithAttentionLSTM(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dims=(64, 128, 256), lstm_hidden_dim=128):
        super(UNet1DWithAttentionLSTM, self).__init__()
        self.encoders = nn.ModuleList()
        self.attention_layers = nn.ModuleList()  # Attention layers in the encoder
        self.decoders = nn.ModuleList()
        self.condition_transform = nn.Linear(condition_dim, hidden_dims[-1])
        self.input_dim = input_dim
        
        # LSTM module for preprocessing
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)
        
        # Encoder layers
        in_channels = 1
        for h_dim in hidden_dims:
            self.encoders.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, h_dim, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2, padding=1)  # Adjust padding
                )
            )
            self.attention_layers.append(SelfAttention(h_dim))  # Add self-attention layer
            in_channels = h_dim

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.bottleneck_attention = SelfAttention(hidden_dims[-1])  # Attention in bottleneck

        # Decoder layers
        for i, h_dim in enumerate(reversed(hidden_dims)):
            self.decoders.append(
                nn.Sequential(
                    nn.ConvTranspose1d(h_dim*2, h_dim//2, kernel_size=2, stride=2),  # Match channels here
                    nn.ReLU()
                )
            )

        # Final layer should produce output with 1 channel
        self.final_layer = nn.Conv1d(hidden_dims[0]//2, 1, kernel_size=1)
        
        # LSTM module for postprocessing (output prediction)
        self.lstm_out = nn.LSTM(lstm_hidden_dim, lstm_hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(lstm_hidden_dim, input_dim)

    def forward(self, x, condition):
        # LSTM preprocessing
        x_lstm, _ = self.lstm(x)  # Apply LSTM to input
        x_lstm = x_lstm.permute(0, 2, 1)  # Adjust shape for further processing

        # Reshape input to [batch_size, 1, input_dim]
        x = x.unsqueeze(1)
        
        # Encoder
        encodings = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            x = x.permute(0, 2, 1) 
            x = self.attention_layers[i](x)  # Apply self-attention
            x = x.permute(0, 2, 1)
            encodings.append(x)

        # Bottleneck
        condition_vector = self.condition_transform(condition).unsqueeze(-1)
        x = self.bottleneck(x + condition_vector)
        x = x.permute(0, 2, 1) 
        x = self.bottleneck_attention(x)  # Apply self-attention in bottleneck
        x = x.permute(0, 2, 1) 

        # Decoder with skip connections
        for i, decoder in enumerate(self.decoders):
            # Skip connection: get the corresponding encoding
            skip_encoding = encodings[-(i+1)]  # Get encoding from corresponding encoder layer
            
            # Ensure skip encoding is the same size as the current x
            if skip_encoding.size(2) != x.size(2):
                skip_encoding = F.interpolate(skip_encoding, size=x.size(2), mode='linear', align_corners=False)

            # Concatenate skip connection with decoder output
            x = torch.cat([x, skip_encoding], dim=1)
            x = decoder(x)

        # Final output: Ensure output has a single channel
        x = self.final_layer(x)

        # Postprocessing with LSTM for better temporal understanding
        x_lstm_out, _ = self.lstm_out(x.permute(0, 2, 1))  # Apply LSTM on output from decoder
        x_lstm_out = x_lstm_out[:, -1, :]  # Get the last output from LSTM
        x = self.fc_out(x_lstm_out)  # Output prediction layer

        return x


class UNet1DWithAttentionLSTM(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dims=(64, 128, 256), lstm_hidden_dim=128):
        super(UNet1DWithAttentionLSTM, self).__init__()
        self.encoders = nn.ModuleList()
        self.attention_layers = nn.ModuleList()  # Attention layers in the encoder
        self.decoders = nn.ModuleList()
        self.condition_transform = nn.Linear(condition_dim, hidden_dims[-1])
        self.input_dim = input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        # LSTM module for preprocessing
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)
        
        # Encoder layers
        in_channels = 1
        for h_dim in hidden_dims:
            self.encoders.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, h_dim, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2, padding=1)  # Adjust padding
                )
            )
            self.attention_layers.append(SelfAttention(h_dim))  # Add self-attention layer
            in_channels = h_dim

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.bottleneck_attention = SelfAttention(hidden_dims[-1])  # Attention in bottleneck

        # Decoder layers
        for i, h_dim in enumerate(reversed(hidden_dims)):
            self.decoders.append(
                nn.Sequential(
                    nn.ConvTranspose1d(h_dim*2, h_dim//2, kernel_size=2, stride=2),  # Match channels here
                    nn.ReLU()
                )
            )

        # Final layer should produce output with 1 channel
        self.final_layer = nn.Conv1d(hidden_dims[0]//2, 1, kernel_size=1)
        
        # LSTM module for postprocessing (output prediction)
        self.lstm_out = nn.LSTM(lstm_hidden_dim, lstm_hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(lstm_hidden_dim, input_dim)

    def forward(self, x, condition):
        x_lstm, _ = self.lstm(x)  # Apply LSTM to input



        # Reshape input to [batch_size, 1, input_dim]
        x = x.unsqueeze(1)
        
        # Encoder
        encodings = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            x = x.permute(0, 2, 1) 
            x = self.attention_layers[i](x)  # Apply self-attention
            x = x.permute(0, 2, 1)
            encodings.append(x)

        # Bottleneck
        condition_vector = self.condition_transform(condition).unsqueeze(-1)
        x = self.bottleneck(x + condition_vector)
        x = x.permute(0, 2, 1) 
        x = self.bottleneck_attention(x)  # Apply self-attention in bottleneck
        x = x.permute(0, 2, 1) 

        # Decoder with skip connections
        for i, decoder in enumerate(self.decoders):
            # Skip connection: get the corresponding encoding
            skip_encoding = encodings[-(i+1)]  # Get encoding from corresponding encoder layer
            
            # Ensure skip encoding is the same size as the current x
            if skip_encoding.size(2) != x.size(2):
                skip_encoding = F.interpolate(skip_encoding, size=x.size(2), mode='linear', align_corners=False)

            # Concatenate skip connection with decoder output
            x = torch.cat([x, skip_encoding], dim=1)
            x = decoder(x)

        # Final output: Ensure output has a single channel
        x = self.final_layer(x)
        x = x.squeeze(1)
        lstmnn = nn.LSTM(x.shape[1], self.lstm_hidden_dim, batch_first=True).cuda()
        # Postprocessing with LSTM for better temporal understanding
        x_lstm_out, _ = lstmnn(x)  # Apply LSTM on output from decoder
        x = self.fc_out(x_lstm_out)  # Output prediction layer

        return x


