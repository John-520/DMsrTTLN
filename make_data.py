# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 16:37:11 2021

@author: 29792
"""
from functools import partial
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import torch.utils.data as data
import numpy as np
import torch
from torchsampler import ImbalancedDatasetSampler



#################################解决num_works != 0时的堵塞问题   没啥用
import cv2
cv2.setNumThreads(0)
#################################



class Mydataset(data.Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.idx = list()
        for item in x:
            self.idx.append(item)
        pass

    def __getitem__(self, index):
        input_data = self.idx[index]
        target = self.y[index]
        return input_data, target

    def __len__(self):
        return len(self.idx)
    def get_labels(self): 
        return self.y

def Make_data(X_train, Y_train, batch_size,shuffle=True):
    datasets = Mydataset(X_train, Y_train)  # 初始化

    dataloader = data.DataLoader(datasets, batch_size=batch_size, shuffle=shuffle, num_workers=0) 
    return dataloader








class Mydataset_multi(data.Dataset):

    def __init__(self, x, y):

        self.x_0 = x[0]["hello"]
        self.x_1 = x[1]["hello"]
        self.x_2 = x[2]["hello"]
        self.x_3 = x[3]["hello"]
        self.x_4 = x[4]["hello"]
        self.x_5 = x[5]["hello"]
        
        self.y = y

        pass

    def __getitem__(self, index):
        input_data_0 = self.x_0[index]
        input_data_1 = self.x_1[index]
        input_data_2 = self.x_2[index]
        input_data_3 = self.x_3[index]
        input_data_4 = self.x_4[index]
        input_data_5 = self.x_5[index]
        
        target = self.y[index]
        
        
        input_data = [
                        {"hello":input_data_0},
                        {"hello":input_data_1},
                        {"hello":input_data_2},
                        {"hello":input_data_3},
                        {"hello":input_data_4},
                        {"hello":input_data_5}
                      ]
        
        
        
        return input_data, target

    def __len__(self):
        return len(self.x_0)
    
    
    def get_labels(self): 
        return self.y

def Make_data_multi(X_train, Y_train, batch_size,shuffle=True):
    datasets = Mydataset_multi(X_train, Y_train)  # 初始化

    dataloader = data.DataLoader(datasets, batch_size=batch_size, shuffle=shuffle, num_workers=0) 
    return dataloader
