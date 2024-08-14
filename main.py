# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 21:34:56 2023

@author: luzy1
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import argparse
import numpy as np
import os

import Shuffle as Shuffle
import make_data
from models import models

import datasets
import sys
import numpy as np
from scipy.io import loadmat
import time
from FocalLoss import FocalLoss 
from ulties import *
from ulties_functions import *
import matplotlib
import matplotlib.pyplot as plt


aa_result = []
for cishu in range(6):
   
                liss = [
                    1,              # 6种时变转速下的跨设备任务
                    2,
                    3,
                    4,
                    5,
                    6,
                    ]   
            

                i = liss[cishu]
                
                aa = []
                bb = []
                cc = []
                aa_processing = []
                aa_loss = []
                
                args = get_args()
                os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
                
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                else:
                    warnings.warn("gpu is not available")
                    device = torch.device("cpu")
                
                    
                SEED = args.seed
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
                os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


                    
                '时变转速下的跨设备研究'
                def load_data(X_train, Y_train, X_test,Y_test,X_test_t,Y_test_t
                              ,batch_size,shuffle=True):
                    loader_src = make_data.Make_data_multi(X_train,Y_train,batch_size,shuffle=shuffle)
                    loader_tar = make_data.Make_data_multi(X_test,Y_test,batch_size,shuffle=shuffle)
                    loader_tar_test = make_data.Make_data_multi(X_test_t,Y_test_t,batch_size,shuffle=shuffle)
                    return loader_src, loader_tar, loader_tar_test


                '数据变量，导入你想要测试的数据'
                # S_train_data, S_train_label, T_train_data,T_train_label, T_test_data, T_test_label,

                dataloaders = load_data(S_train_data, S_train_label, T_train_data,T_train_label,
                                        T_test_data,
                                        T_test_label,
                                    args.batch_size,shuffle=True)
                    
            
            
                
                model = models(args)
                model.to(device)
                
                        
                def weights_init_kaiming(m):
                    classname = m.__class__.__name__
                    if classname.find('Conv1d') != -1:
                        torch.nn.init.kaiming_uniform_(m.weight)
                        if m.bias is not None:
                            torch.nn.init.zeros_(m.bias)                   ###
                    elif classname.find('BatchNorm1d') != -1:
                        torch.nn.init.normal_(m.weight, 1.0, 0.02)
                        torch.nn.init.zeros_(m.bias)
                        
                    elif classname.find('Linear') != -1:
                        torch.nn.init.xavier_normal_(m.weight)
                        if m.bias is not None:
                            torch.nn.init.zeros_(m.bias)
    
                
                model.apply(weights_init_kaiming)
                
                
          
    
                correct = 0
                stop = 0
                
                
            
                import torch
                from RAdam import RAdam

                optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=args.decay) #
                time_start = time.time() #开始计时
                
                
                
                label_cha = 1
                labels = 1
                epoch_epoch = args.nepoch
                for epoch in range(1, args.nepoch + 1):
                    stop += 1
                    
                
                    '训练和测试环节'
                    train_loss_list, train_acc_sum = train_epoch(labels, label_cha,epoch_epoch, epoch, model, dataloaders, optimizer,i)
                    print(f'Epoch: [{epoch:2d}]')
                    t_correct = test(model, dataloaders[-1])
                    
                    
                    '保存训练损失和测试精度结果'
                    a_processing = 100. * t_correct / len(dataloaders[-1].dataset)
                    aa_processing.append(a_processing)
                    a_loss = sum(train_loss_list).cpu().data.numpy()
                    aa_loss.append(a_loss)
                    
                    
                    
                    torch.save(model, 'pseudo_model.pkl')    
                    if t_correct > correct:
                        correct = t_correct
                        stop = 0 
                        torch.save(model, 'model.pkl')
                    print(
                        f'{args.src}-{args.tar}: max correct: {correct} max accuracy: {100. * correct / len(dataloaders[-1].dataset):.2f}%\n')
            
                    if stop >= args.early_stop:
                        print(
                            f'Final test acc: {100. * correct / len(dataloaders[-1].dataset):.2f}%')
                        break
                    
                    
                    
                time_end = time.time()    #结束计时
                time_c= time_end - time_start   #运行所花时间
                print('time cost', time_c, 's')

                
