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





class GNet_multihead_attention(nn.Module):
    def __init__(self, input_dim, hidden_size=32, alpha=0.5, num_heads=4):
        super(GNet_multihead_attention, self).__init__()
        self.alpha = alpha
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.relu = nn.Tanh()
        self.norm = nn.LayerNorm(hidden_size)
        
        # Fully connected layers
        self.fc0 = nn.Linear(input_dim+1, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size*self.num_heads, hidden_size)
        self.fc_out1 = nn.Linear(hidden_size, hidden_size)
        self.z_out = nn.Linear(hidden_size, 1)
        
        # Multi-head self-attention layers for `a`
        self.attention_query = nn.Linear(1, hidden_size * num_heads)
        self.attention_key = nn.Linear(1, hidden_size * num_heads)
        self.attention_value = nn.Linear(1, hidden_size * num_heads)
        self.softmax = nn.Softmax(dim=-1)

        
        self.embedding = nn.EmbeddingBag(100, 16, mode='mean')
        self.fc_emb = nn.Linear(16, hidden_size)
        
        
    def forward(self, x, a,ddpmoutput):
        # Process input `x`
        out1 =  self.relu(torch.cat([x,ddpmoutput],1))
        out1 = self.relu(self.fc0(out1))
        out1 = self.relu(self.fc1(out1))
        out1 = self.relu(self.fc2(out1))
        
        # Multi-head self-attention for `a`
        query = self.attention_query(a).view(-1, self.num_heads, self.hidden_size)
        key = self.attention_key(a).view(-1, self.num_heads, self.hidden_size)
        value = self.attention_value(a).view(-1, self.num_heads, self.hidden_size)
        
        # Compute attention scores and apply to values
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        attention_weights = self.softmax(attention_scores)
        out_z = torch.matmul(attention_weights, value).view(-1, self.hidden_size * self.num_heads)
        out_z = self.relu(self.fc3(out_z))
        
        
        a_emb = self.embedding(a.long())
        a_emb = self.relu(self.fc_emb(a_emb))
        
        
        
        # Combine results
        out = out1 + out_z+a_emb
        out = self.fc_out1(out)
        out = self.z_out(self.relu(out))
        
        return out
    


    def loss_fn_2(self, y_pred, y_pred2, y_true):
        Out_G = y_pred
        Out_G2 = y_pred2
        fac = 2
        dist = torch.abs(Out_G - Out_G2)
        sd_y = torch.std(y_true).item()
        pen = torch.log((dist / sd_y) + 1.0 / fac)
        return pen.mean()
    


    def loss_fn(self, y_true,con_x,ddpmoutput):

        
        y_pred = self.forward(con_x, self.alpha * torch.ones(y_true.shape[0], 1).cuda(),ddpmoutput)
        pinball_loss = torch.mean(torch.max(self.alpha * (y_true - y_pred), (self.alpha - 1) * (y_true - y_pred)))
        rand_aray = torch.rand_like(y_true).cuda()
        y_pred2 = self.forward(con_x, rand_aray,ddpmoutput)
        loss3 = self.loss_fn_2(y_pred, y_pred2, y_true)
        pinball_loss -= loss3
        return pinball_loss 


    
    
    




    def trainGmul(x_train,y_train,x_val,y_val,alpha,ifattention = False):
        if ifattention:
            qmodel = GNet_multihead_attention(input_dim,hidden_size=32,alpha = alpha).cuda()
        else:
            qmodel = GNet0(input_dim,hidden_size=32,alpha = alpha).cuda()
        # qmodel = GNet0(input_dim,hidden_size=32,alpha = alpha).cuda()
        optimizer = optim.Adam(qmodel.parameters(), lr=0.001)


        # 假设验证集是 x_val, y_val
        early_stopping_patience = 500  # 如果连续 100 个 epoch 验证损失不降低，则停止训练
        best_loss = float('inf')  # 用于记录验证集上的最优损失
        patience_counter = 0  # 记录验证损失未降低的连续 epoch 数

        num_epochs = 2000
        for epoch in range(num_epochs):
            qmodel.train()  # 设置模型为训练模式
            optimizer.zero_grad()  # 清零梯度
            
            # 训练集前向传播和损失计算
            rand_aray = torch.rand_like(y_train).cuda()
            y_pred = qmodel(x_train, alpha * torch.ones(x_train.shape[0], 1).cuda())
            y_pred2 = qmodel(x_train, rand_aray)
            loss = qmodel.loss_fn(y_pred, y_train)
            loss3 = qmodel.loss_fn_2(y_pred, y_pred2, y_train)
            loss -= loss3
            loss.backward()
            optimizer.step()
            
            # 验证集前向传播和损失计算
            qmodel.eval()  # 设置模型为评估模式
            with torch.no_grad():
                val_pred = qmodel(x_val, alpha * torch.ones(x_val.shape[0], 1).cuda())
                val_loss = qmodel.loss_fn(val_pred, y_val)

            # Early Stopping 检测
            if val_loss.item() < best_loss:
                best_loss = val_loss.item()  # 更新最优验证损失
                patience_counter = 0  # 重置计数器
                # 可以保存当前最优模型
                torch.save(qmodel.state_dict(), "best_model.pth")
            else:
                patience_counter += 1  # 增加计数器

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            # 每隔 10 个 epoch 打印一次训练和验证损失
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")


