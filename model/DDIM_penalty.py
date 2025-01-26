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





class con_Backbone(nn.Module):
    def __init__(self, n_steps, input_dim = 1,con_dim=4):
        super().__init__()
        self.linear_model1 = nn.Sequential(
            nn.Linear(input_dim+con_dim+1, 32),
            nn.ReLU()
        )
        # Condition time t
        self.embedding_layer = nn.Embedding(n_steps, 32)
        
        self.linear_model2 = nn.Sequential(
            nn.Linear(32+con_dim, 32),
            nn.ReLU(),
            
            nn.Linear(32, 64),
            nn.ReLU(),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, input_dim),
        )
    def forward(self, x, con_x,alpha,idx):   
        x = torch.cat((x, con_x), dim=1)
        x= torch.cat((x,alpha),dim=1) 
        x = torch.cat((self.linear_model1(x)+self.embedding_layer(idx),con_x),dim = 1)
        x = self.linear_model2(x)
        return x


class DDIM_penalty(nn.Module):
    def __init__(self, device, beta_1, beta_T, T, input_dim ,con_dim,scheduling = 'uniform',tau = 1,alpha = 0.1):
        '''
        The epsilon predictor of diffusion process.

        beta_1    : beta_1 of diffusion process
        beta_T    : beta_T of diffusion process
        T         : Diffusion Steps
        input_dim : a dimension of data

        '''
        super().__init__()
        self.device = device
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start = beta_1, end=beta_T, steps=T), dim = 0).to(device = device)
        self.backbone = con_Backbone(T, input_dim,con_dim).to(device = device)
        self.scheduling = scheduling
        self.tau = min(tau, T)
        self.input_dim = input_dim
        self.to(device = self.device)
        self.alpha = alpha

    def loss_fn(self, x,con_x, idx=None,quantile_list = np.linspace(0, 1, 11)):

        quantile_loss = 0
        output, epsilon, alpha_bar = self.forward(x,con_x, idx=idx, get_target=True)
        loss = (output - epsilon).square().mean()
        ddpmoutput = self.sampling(x.shape[0],con_x, only_final=True)
        weight_list = np.exp(-np.abs(quantile_list - self.alpha))
        weight_list = weight_list / weight_list.sum()
        weight_list = torch.tensor(weight_list).cuda()
        
        # 计算 quantile_loss
        quantile_value = torch.tensor(np.percentile(ddpmoutput.detach().cpu().numpy(), quantile_list)).cuda()
        y_pred1 = x.detach().cpu().numpy()
        quantile_predic = torch.tensor(np.percentile(y_pred1, quantile_list)).cuda()
        quantile_loss = torch.abs(quantile_value - quantile_predic)
        quantile_loss_std = torch.std(quantile_loss) + 1e-6  # 防止除以零
        quantile_loss_normalized = quantile_loss / quantile_loss_std
        quantile_loss = torch.mean(weight_list * quantile_loss_normalized)
        scaling_factor = quantile_loss / (quantile_loss + loss)
        dynamic_weight = 0.1 * scaling_factor  # 基于比例动态调整

        return loss + dynamic_weight*quantile_loss
    

        
    def forward(self, x,con_x, idx=None, get_target=False):
        '''
        x          : real data if idx==None else perturbation data
        idx        : if None (training phase), we perturbed random index. Else (inference phase), it is recommended that you specify.
        get_target : if True (training phase), target and sigma is returned with output (epsilon prediction)

        '''

        if idx == None:
            idx = torch.randint(0, len(self.alpha_bars), (x.size(0), )).to(device = self.device)
            used_alpha_bars = self.alpha_bars[idx][:, None]
            epsilon = torch.randn_like(x)
            x_tilde = torch.sqrt(used_alpha_bars) * x + torch.sqrt(1 - used_alpha_bars) * epsilon
            
        else:
            idx = torch.Tensor([idx for _ in range(x.size(0))]).to(device = self.device).long()
            x_tilde = x
            
        alphal = torch.Tensor([self.alpha for _ in range(x.size(0))]).to(device = self.device).reshape(-1,1)    
            
        output = self.backbone(x_tilde, con_x,alphal,idx)
        
        return (output, epsilon, used_alpha_bars) if get_target else output
    

        
    def _get_process_scheduling(self, reverse = True):
        if self.scheduling == 'uniform':
            diffusion_process = list(range(0, len(self.alpha_bars), self.tau)) + [len(self.alpha_bars)-1]
        elif self.scheduling == 'exp':
            diffusion_process = (np.linspace(0, np.sqrt(len(self.alpha_bars)* 0.8), self.tau)** 2)
            diffusion_process = [int(s) for s in list(diffusion_process)] + [len(self.alpha_bars)-1]
        else:
            assert 'Not Implementation'
            
        
        diffusion_process = zip(reversed(diffusion_process[:-1]), reversed(diffusion_process[1:])) if reverse else zip(diffusion_process[1:], diffusion_process[:-1])
        return diffusion_process
            

    
    def _one_reverse_diffusion_step(self, x,con_x):
        '''
        x   : perturbated data (输入被加噪的样本)
        '''
        diffusion_process = self._get_process_scheduling(reverse=True)  # 获取时间步调度
        for prev_idx, idx in diffusion_process:
            predict_epsilon = self.forward(x, con_x,idx)
            
            predicted_x0 =  (
                (x - torch.sqrt(1 - self.alpha_bars[idx]) * predict_epsilon) 
                / torch.sqrt(self.alpha_bars[idx])
            )
            
            # 确定性更新方向
            direction_pointing_to_xt = torch.sqrt(1 - self.alpha_bars[prev_idx]) * (predict_epsilon)
            
            x = torch.sqrt(self.alpha_bars[prev_idx]) *predicted_x0 + direction_pointing_to_xt

            yield x

    @torch.no_grad()
    def sampling(self, sampling_number, con_x,sample=None, only_final=True):
        '''
        sampling_number : a number of generation
        only_final      : If True, return is an only output of final schedule step 
        '''
        if sample==None:
            sample = torch.randn([sampling_number,self.input_dim]).to(device = self.device)
            
        sampling_list = []
        final = None
        for sample in self._one_reverse_diffusion_step(sample,con_x):
            final = sample
            if not only_final:
                sampling_list.append(final)

        return final if only_final else torch.stack(sampling_list)

    def train(self, optimizer, num_epochs, targetdim, traindata_loader, valdata_loader, early_stopping, model_save_path='best_ddim_model.pth'):
        best_loss = float('inf')
        early_stopping_counter = 0
        for epoch in range(num_epochs):
            whole_loss = 0
            for i, batch in enumerate(traindata_loader):
                batch_size = batch.shape[0]
                if targetdim == 1:
                    batch = batch.cuda()
                    y1 = batch[:, -1].reshape(-1, 1).cuda()
                    x1 = batch[:, :-1].cuda()
                else:
                    batch = batch.cuda()
                    y1 = torch.Tensor(batch[:, -targetdim:]).cuda()
                    x1 = batch[:, :-targetdim].cuda()
                loss = self.loss_fn(y1, x1)
                whole_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            val_loss = 0
            with torch.no_grad():
                for val_batch in valdata_loader:
                    if targetdim == 1:
                        batch = val_batch.cuda()
                        y1 = batch[:, -1].reshape(-1, 1).cuda()
                        x1 = batch[:, :-1].cuda()
                    else:
                        batch = val_batch.cuda()
                        y1 = torch.Tensor(batch[:, -targetdim:]).cuda()
                        x1 = batch[:, :-targetdim].cuda()
                    val_loss += self.loss_fn(y1, x1)
                val_loss /= len(valdata_loader)
            if (epoch) % 20 == 0:
                print('epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch, whole_loss / len(traindata_loader), val_loss.item()))
            loss_new = val_loss
            if loss_new < best_loss:
                best_loss = loss_new
                early_stopping_counter = 0
                print('epoch: {}, find new best loss: Train Loss: {:.4f}'.format(epoch, best_loss))
                print('-' * 10)
                # Save the best model
                torch.save(self.state_dict(), model_save_path)
            else:
                early_stopping_counter += 1
            if early_stopping_counter == early_stopping:
                print("Early stopping after {} epochs".format(epoch))
                break





