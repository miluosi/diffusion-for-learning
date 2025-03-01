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
            nn.Linear(input_dim+con_dim, 32),
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
    def forward(self, x, con_x,idx):   
        x = torch.cat((x, con_x), dim=1) 
        x = torch.cat((self.linear_model1(x)+self.embedding_layer(idx),con_x),dim = 1)
        x = self.linear_model2(x)
        return x
    


class con_DDPM(nn.Module):
    def __init__(self, device, beta_1, beta_T, T, input_dim,con_dim):
        '''
        The epsilon predictor of diffusion process.

        beta_1    : beta_1 of diffusion process
        beta_T    : beta_T of diffusion process
        T         : Diffusion Steps
        input_dim : a dimension of data

        '''
        super().__init__()
        self.betas = torch.linspace(start = beta_1, end=beta_T, steps=T)
        self.alphas = 1 - self.betas
        self.device = device
        self.input_dim = input_dim
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start = beta_1, end=beta_T, steps=T), dim = 0).to(device = device)
        self.con_dim = con_dim
        self.backbone = con_Backbone(T, input_dim,con_dim)
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=device), self.alpha_bars[:-1]])
        self.step_count = 0
        self.to(device = self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def _one_diffusion_step(self, x,con_x):
        '''
        x   : perturbated data
        '''
        for idx in reversed(range(len(self.alpha_bars))):
            noise = torch.zeros_like(x) if idx == 0 else torch.randn_like(x)
            sqrt_tilde_beta = torch.sqrt((1 - self.alpha_prev_bars[idx]) / (1 - self.alpha_bars[idx]) * self.betas[idx])
            idx1 = torch.Tensor([idx for _ in range(x.size(0))]).to(device = self.device).long()
            predict_epsilon = self.backbone(x,con_x, idx1)
            mu_theta_xt = torch.sqrt(1 / self.alphas[idx]) * (x - self.betas[idx] / torch.sqrt(1 - self.alpha_bars[idx]) * predict_epsilon)
            x = mu_theta_xt + sqrt_tilde_beta * noise
            yield x


    def _one_diffusion_step_storerand(self, x, con_x):
        for idx in reversed(range(len(self.alpha_bars))):
            noise = torch.zeros_like(x) if idx == 0 else torch.randn_like(x)
            sqrt_tilde_beta = torch.sqrt((1 - self.alpha_prev_bars[idx]) / (1 - self.alpha_bars[idx]) * self.betas[idx])
            idx1 = torch.Tensor([idx for _ in range(x.size(0))]).to(device=self.device).long()
            predict_epsilon = self.backbone(x, con_x, idx1)
            mu_theta_xt = torch.sqrt(1 / self.alphas[idx]) * (x - self.betas[idx] / torch.sqrt(1 - self.alpha_bars[idx]) * predict_epsilon)
            x = mu_theta_xt + sqrt_tilde_beta * noise
            yield x, noise  # Yield x and noise directly as two separate values

    
    
    def loss_fn(self, x,con_x, idx=None,quantile_list = np.linspace(0, 1, 11)):
        self.step_count += 1
        quantile_loss = 0
        output, epsilon, alpha_bar = self.forward(x,con_x, idx=idx, get_target=True)
        loss = (output - epsilon).square().mean()

        return loss 



    @torch.no_grad()
    def sampling(self, sampling_number,con_x, only_final=False,ifreturnrand = False):
        '''
        sampling_number : a number of generation
        only_final      : If True, return is an only output of final schedule step 
        '''
        sample = torch.randn([sampling_number,self.input_dim]).to(device = self.device).squeeze().reshape(-1,1)
        sampling_list = []
        rand_list = []
        final = None
        if not ifreturnrand:
            for idx, sample in enumerate(self._one_diffusion_step(sample,con_x)):
                final = sample
                if not only_final:
                    sampling_list.append(final)
            return final if only_final else torch.stack(sampling_list)
        else:
            for idx, (sample, noise) in enumerate(self._one_diffusion_step_storerand(sample, con_x)): # Unpack x and noise directly
                final = sample
                rand_list.append(noise)
                if not only_final:
                    sampling_list.append(final)

            return final, torch.stack(rand_list) if only_final else torch.stack(sampling_list), torch.stack(rand_list)
    
    
        
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
            

            
        output = self.backbone(x_tilde,con_x, idx)
        
        return (output, epsilon, used_alpha_bars) if get_target else output
    
    def train(self, num_epochs, targetdim, traindata_loader, valdata_loader, early_stopping, model_save_path='best_ddpm_model.pth'):
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
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
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
    


    
