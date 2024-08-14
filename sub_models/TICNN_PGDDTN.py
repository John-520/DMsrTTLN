# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 23:16:48 2023

@author: luzy1
"""
from torch import nn
import warnings
import torch
import numpy as np
import torch.nn.functional as F
import sub_models
import sys 
sys.path.append("D:\北京交通大学博士\论文【小】\论文【第五章】\code\Methods\SPCNN") 

'行列正则化'
def l2row_torch(X):
	"""
	L2 normalize X by rows. We also use this to normalize by column with l2row(X.T)
	"""
	N = torch.sqrt((X**2).sum(axis=1)+1e-8)
	Y = (X.T/N).T
	return Y,N



class TICNN_PGDDTN(nn.Module):
    def __init__(self, args, pretrained=False,  in_channel=1, out_channel=5):
        super(TICNN_PGDDTN, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")


        self.args = args
        self.resnet_features_1d = getattr(sub_models, 'resnet_features_1d')()
        self.__in_features = 512 
        
        
    def forward(self, x):
        
        
        
        xxxxx = x

        '字典中的数据提取'
        x_0 = x[0]["hello"].cuda()    # 振动信号
        x_1 = x[1]["hello"].cuda()    #转速信号       
        x_2 = x[2]["hello"].cuda()    #相位函数      用于广义解调
        x_3 = x[3]["hello"].cuda()    #外圈故障特征系数
        x_4 = x[4]["hello"].cuda()    #内圈故障特征系数
        x_5 = x[5]["hello"].cuda()    #滚子故障特征系数
        
        x = x_0

        x = self.resnet_features_1d(x)
        
        x = x.view(x.size(0), -1)
        x = torch.sqrt(x**2 + 1e-8)
        x = l2row_torch(x)[0]
        x = l2row_torch(x.T)[0].T        
        
        x = x.view(x.size(0), -1)
        
        return x



    def output_num(self):
        return self.__in_features









