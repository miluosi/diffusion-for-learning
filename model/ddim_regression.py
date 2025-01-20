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

from DDIM_penalty import DDIM_penalty
from GNet_multihead_attention import GNet_multihead_attention



class DDIM_regression(nn.Module):
    def __init__(self, device, beta_1, beta_T, T, input_dim,con_dim,alpha = 0.1,tau = 1):
        '''
        The epsilon predictor of diffusion process.

        beta_1    : beta_1 of diffusion process
        beta_T    : beta_T of diffusion process
        T         : Diffusion Steps
        input_dim : a dimension of data

        '''
        super().__init__()
        self.device = device
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T
        self.input_dim = input_dim
        self.con_dim = con_dim
        self.ddpm = DDIM_penalty(device, beta_1, beta_T, T, input_dim,con_dim,scheduling="uniform",tau=tau,alpha=alpha).to(device)
        self.gnet = GNet_multihead_attention(con_dim, hidden_size=32, alpha=alpha, num_heads=4).to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)



    def ddpmsampling(self, x,con_x):
        ddpmoutput = self.ddpm.sampling(x.shape[0],con_x, only_final=True)
        return ddpmoutput

    def forward(self, x,con_x):
        ddpmoutput = self.ddpm.sampling(x.shape[0],con_x, only_final=True)
        gnetoutput = self.gnet(con_x, self.alpha * torch.ones(x.shape[0], 1).to(self.device),ddpmoutput)
        return gnetoutput
    
    def loss_fn(self, x, con_x, lambda1, alpha, idx=None, train_stage="joint"):

        
        # 根据 train_stage 决定冻结哪一部分网络的梯度
        if train_stage == "quantile":
            # 冻结 diffusion 部分
            for param in self.ddpm.parameters():
                param.requires_grad = False
            for param in self.gnet.parameters():
                param.requires_grad = True
            ddpmoutput = self.ddpm.sampling(x.shape[0],con_x, only_final=True).to(self.device)
            quantile_loss = self.gnet.loss_fn(x, con_x,ddpmoutput)
            return lambda1*quantile_loss

        elif train_stage == "diffusion":
            # 冻结 quantile 部分
            for param in self.gnet.parameters():
                param.requires_grad = False
            for param in self.ddpm.parameters():
                param.requires_grad = True
            ddpmoutput = self.ddpm.sampling(x.shape[0],con_x, only_final=True).to(self.device)
            ddpm_loss = self.ddpm.loss_fn(x, con_x)
            return ddpm_loss

        elif train_stage == "joint":
            # 两部分都更新
            for param in self.ddpm.parameters():
                param.requires_grad = True
            for param in self.gnet.parameters():
                param.requires_grad = True
            
            ddpmoutput = self.ddpm.sampling(x.shape[0],con_x, only_final=True).to(self.device)
            ddpm_loss = self.ddpm.loss_fn(x, con_x)
            quantile_loss = self.gnet.loss_fn(x, con_x,ddpmoutput)
            return lambda1 * quantile_loss + (1-lambda1) * ddpm_loss

        else:
            raise ValueError("Invalid train_stage. Choose from ['quantile', 'diffusion', 'joint'].")
    

    def train(self,num_epochs,traindata_loader,valdata_loader,targetdim = 1,early_stopping = 500):
        best_loss = float('inf')
        early_stopping_counter = 0
        for epoch in range(num_epochs):
            whole_loss = 0
            for i, batch in enumerate(traindata_loader):
                batch_size = batch.shape[0]
                if targetdim==1:
                    batch = batch.cuda()
                    y1 = batch[:,-1].reshape(-1,1).cuda()
                    x1 = batch[:,:-1].cuda()
                else:
                    batch = batch.cuda()
                    y1 =  torch.Tensor(batch[:,-targetdim:]).cuda() 
                    x1 = batch[:,:-targetdim].cuda()


                loss = self.loss_fn(y1, x1, lambda1=0.0, alpha=self.alpha, train_stage="diffusion")
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()



                loss = self.loss_fn(y1,x1, lambda1=1.0, alpha=self.alpha, train_stage="quantile")
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                
                loss = self.loss_fn(y1, x1, lambda1=0.5, alpha=self.alpha, train_stage="joint")
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            val_loss = 0
            with torch.no_grad():
                for val_batch in valdata_loader:
                    if targetdim==1:
                        batch = val_batch.cuda()
                        y1 = batch[:,-1].reshape(-1,1).cuda()
                        x1 = batch[:,:-1].cuda()
                    else:
                        batch = val_batch.cuda()
                        y1 =  torch.Tensor(batch[:,-targetdim:]).cuda() 
                        x1 = batch[:,:-targetdim].cuda()
                    val_loss+=self.loss_fn(y1,x1,lambda1=0.5, alpha=self.alpha, train_stage="joint")
                val_loss /= len(valdata_loader)
            if (epoch) % 20 == 0:
                print('epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch, whole_loss/len(traindata_loader), val_loss.item()))
            loss_new = val_loss
            if loss_new < best_loss:
                best_loss = loss_new
                early_stopping_counter = 0
                print('epoch: {}, find new best loss: Train Loss: {:.4f}'.format(epoch,best_loss))
                print('-' * 10)
                torch.save(self.ddpm.state_dict(), 'ddpm_model.pth')
                torch.save(self.gnet.state_dict(), 'gnet_model.pth')
            else:
                early_stopping_counter += 1
            if early_stopping_counter == early_stopping:
                print("Early stopping after {} epochs".format(epoch))
                break