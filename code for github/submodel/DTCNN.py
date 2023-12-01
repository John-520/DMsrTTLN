# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 18:00:38 2023

@author: luzy1
"""

from torch import nn
import warnings

class DTCNN(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, out_channel=5):
        super(DTCNN, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")

        self.feature_layers = nn.Sequential(
            
            ########   81 = 5+4*19   ;11+10*7;    21+20*3
            nn.Conv1d(1, 16, kernel_size=81, stride=8,padding=1,dilation=1),  # 16, 26 ,26  ###105###   nn.Conv1d(1, 16, kernel_size=81, stride=8,padding=1,dilation=1), 
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.MaxPool1d(kernel_size=2, stride=2,padding=1),
            
            nn.Conv1d(16, 32, kernel_size=3, stride=1,padding=1),  # 32, 24, 24
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(32, 64, kernel_size=3, stride=1,padding=1),  # 32, 24, 24
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(64, 64, kernel_size=3, stride=1,padding=1),  # 32, 24, 24
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(64, 64, kernel_size=3, stride=1,padding=1),  # 32, 24, 24
            nn.BatchNorm1d(64),
            # nn.RReLU(0.0,0.5),
            nn.ReLU(inplace=True),
          #  nn.SELU(inplace=True),
            # nn.Dropout(0.5),     
            nn.MaxPool1d(kernel_size=2, stride=2)
            )
        self.__in_features = 256   #576    256
        
    def forward(self, x):
        x = self.feature_layers(x)
        return x


    def output_num(self):
        return self.__in_features






