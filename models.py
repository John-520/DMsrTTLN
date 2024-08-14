# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 19:12:18 2021

@author: 29792
"""
import numpy as np
import torch
import torch.nn as nn
from loss.MMD import mmd_rbf_noaccelerate, mmd_rbf_accelerate
from loss.KMMD import kmmd_loss
import sys 


import sub_models
import torch.nn.functional as F

from sklearn.cluster import KMeans
from loss.adv import *




'行列正则化'
def l2row_torch(X):
	"""
	L2 normalize X by rows. We also use this to normalize by column with l2row(X.T)
	"""
	N = torch.sqrt((X**2).sum(axis=1)+1e-8)
	Y = (X.T/N).T
	return Y,N
 

    
class SwiGLU(nn.Module):
    def __init__(self, dim=256, hidden_dim=512, multiple_of=2, dropout=0.5):
        super().__init__()
        hidden_dim = 256
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.w3 = nn.Linear(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
    

def Kurtosis_activation(x, x_f):
    x = x.view(x.size(0), -1)
    x_f = x_f.view(x_f.size(0), -1)
    
    '随机选择峰峰值的前几个值'
    # 计算振动信号的峰峰值
    x_kurtosis = torch.max(x, dim=1).values  - torch.min(x, dim=1).values  
    num_positions = torch.randint(10, 20, (1,)).item()  # 
    sorted_indices = torch.argsort(x_kurtosis.squeeze())  # 
    selected_indices = sorted_indices[:num_positions]  # 
    x_f[selected_indices] *= 0.05        #

    return x_f
    


class models(nn.Module):

    def __init__(self, args):
        super(models, self).__init__()
        self.args = args
        
        self.num_classes= args.num_classes
        self.feature_layers = getattr(sub_models, args.model_name)(args, args.pretrained)

        self.bottle = nn.Sequential(nn.Linear(self.feature_layers.output_num(), 256)) #100   512   64     512 , nn.ReLU(), nn.Dropout()
        self.drop = nn.Dropout(0)

        self.cls_fc = nn.Linear(256, self.num_classes) #

            
            
        self.SwiGLU = nn.GELU()
        self.Dropout = nn.Dropout()   # 0.5
        
        self.Kurtosis_activation = Kurtosis_activation
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
        
        



    def l2_zheng(self,x):
        x = torch.sqrt(x**2 + 1e-8)
        x = l2row_torch(x)[0]
        x = l2row_torch(x.T)[0].T        
        return x


    def forward(self, source,target,label_source,label_target,epoch,mu_value,task): #(64,1,1200)
        loss = 0
        adv_loss=0
        dist_loss_kmmd = 0
        
        
        
        source_source = source[0]["hello"].cuda()
        target_target = target[0]["hello"].cuda()
        
        
        source = self.feature_layers(source)           #
        f_source = source.view(source.size(0), -1)
        source = self.bottle(source)
        source = self.SwiGLU(source)
        source = self.Dropout(source)
    
        source = self.l2_zheng(source)
        s_pred = self.cls_fc(source)
        

        



        target = self.feature_layers(target)
        f_target = target.view(target.size(0), -1)
        target = self.bottle(target)
        target = self.SwiGLU(target)
        target = self.Dropout(target)
        
        target = self.l2_zheng(target)
        target = self.Kurtosis_activation(target_target,target)
        t_pred = self.cls_fc(target)
        

        if self.training == True and epoch> self.args.middle_epoch:               
        # 定义对抗损失函数
            if self.args.ADNN:
                net = AdversarialLoss()
                adv_loss +=  net(f_source,f_target)
            else:
                adv_loss = 0

            distance_loss = kmmd_loss
            dist_loss_kmmd += distance_loss(source,
                                            target, 
                                            source_source,
                                            target_target,
                                            )

            loss =   dist_loss_kmmd
            
            
            '动态MMD和DAN损失之间的权重'
            kl = F.kl_div(f_target.softmax(dim=-1).log(), f_source.softmax(dim=-1), reduction='sum')

            if mu_value == 2:
                with open('results_miu.txt','a') as file0:
                    print([task],mu_value,np.mean(1-kl.item()),np.mean(kl.item()),file=file0)
                    

        return s_pred, t_pred,  (1-kl) * loss, kl * adv_loss 



    def predict(self, x):
        x = self.feature_layers(x)
        x = self.bottle(x)
        x = self.l2_zheng(x)

        return self.cls_fc(x)



    def predict_fea(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        x = self.bottle(x)


        x = self.cls_fc(x)
        return x
    
