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

class self_attention(nn.Module):
    def __init__(self, input_dim, hidden_size=32, alpha=0.1, num_heads=4):
        super(self_attention, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.W_q = nn.Linear(input_dim, hidden_size)
        self.W_k = nn.Linear(input_dim, hidden_size)
        self.W_v = nn.Linear(input_dim, hidden_size)
        self.W_o = nn.Linear(hidden_size, input_dim)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=1)
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x, alpha, ddpmoutput):
        batch_size = x.shape[0]
        q = self.W_q(x).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.W_k(x).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.W_v(x).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attention = self.softmax(torch.matmul(q, k.permute(0, 1, 3, 2)) / np.sqrt(self.head_dim))
        attention = self.dropout(attention)
        out = torch.matmul(attention, v).permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.hidden_size)
        out = self.W_o(out)
        out = self.layer_norm(out + x)
        return out


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

    
    