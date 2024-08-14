# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:06:33 2022

@author: 29792
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import argparse
import numpy as np
import os

#import data_loader
import Shuffle as Shuffle
import make_data
import make_unbalanced_data
import mmd
from models import models

import datasets
import sys
import numpy as np
from scipy.io import loadmat
import time
from FocalLoss import FocalLoss 
from ulties import *




# Consider the gpu or cpu condition
if torch.cuda.is_available():
    device = torch.device("cuda")
    # device = torch.device("cpu")
else:
    warnings.warn("gpu is not available")
    device = torch.device("cpu")
                    
                    
import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from operator import truediv
from sklearn.utils.multiclass import unique_labels



from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score

            

def load_data(X_train, Y_train, X_test,Y_test,X_test_t,Y_test_t
              ,batch_size,shuffle=True):
    loader_src = make_data.Make_data(X_train,Y_train,batch_size,shuffle=shuffle)
    loader_tar = make_data.Make_data(X_test,Y_test,batch_size,shuffle=shuffle)
    loader_tar_test = make_data.Make_data(X_test_t,Y_test_t,batch_size,shuffle=shuffle)
    return loader_src, loader_tar, loader_tar_test



from timm.loss import LabelSmoothingCrossEntropy

def train_epoch(labels, label_cha, epoch_epoch, epoch, model, dataloaders, optimizer,task):
    
    train_loss_list,train_acc_list= [],[]
    acc_sum =0.0
    n = 0
    dist_loss_shang = 0

    
    model.train()
    source_loader, target_train_loader, _ = dataloaders
    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)

    num_iter = len(source_loader)
    num_iter_target = len(target_train_loader)
    for i in range(1, num_iter):
        data_source, label_source = next(iter_source)  #一个个取值
        data_target, label_target = next(iter_target)
        
        label_source =  label_source.to(device)
        label_target =  label_target.to(device)
        
        if i % num_iter_target == 0:
            iter_target = iter(target_train_loader)
        if i % num_iter == 0:
            iter_source = iter(source_loader)

        '训练过程'
        optimizer.zero_grad()

        if i == num_iter-1:
            mu_value = 2
        else:
            mu_value = 0

        label_source_pred,label_target_pred, loss_mmd, adversarial_loss = model(data_source, data_target,label_source,label_target,epoch,mu_value,task)
        loss_cls = LabelSmoothingCrossEntropy(smoothing=0.3)(label_source_pred, label_source)

        loss =  loss_cls   +  1 *  loss_mmd   - 1 * adversarial_loss             #

        loss.backward()
        optimizer.step()
        
        if epoch < args.middle_epoch+1:     #
            loss_mmd, adversarial_loss= torch.tensor(0),torch.tensor(0)

        #导出loss和acc
        acc_sum += (label_source_pred.argmax(dim=1)== label_source).sum()
        n +=label_source.shape[0]
        acc = acc_sum/n
        train_loss_list.append(loss)
        train_acc_list.append(acc)
    return train_loss_list, train_acc_list






def test(model, dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            target = target.cuda()
            pred = model.predict(data)
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()
            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(dataloader)
        print(
            f'Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(dataloader.dataset)} ({100. * correct / len(dataloader.dataset):.2f}%)')
    return correct




def get_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, help='Root path for dataset',
                        default='/data/zhuyc/OFFICE31/')
    parser.add_argument('--src', type=str,
                        help='Source domain', default='设备A')
    parser.add_argument('--tar', type=str, 
                        help='Target domain', default='设备B')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    
    
    
    parser.add_argument('--model_name', type=str, default='TICNN_PGDDTN', help='the name of the model')  #DASAN DCTLN  MIXCNN  TICNN_5   resnet_features_1d   resnet_features_1d
    parser.add_argument("--pretrained", type=bool, default=False, help='whether to load the pretrained model')
    #显示距离域损失
    parser.add_argument('--distance_metric', type=bool, default=False, help='whether use distance metric')
    parser.add_argument('--distance_loss', type=str, choices=['MK-MMD','MMD', 'JMMD', 'CORAL','LMMD','CKB+MMD'], 
                        default=False, help='which distance loss you use')    #CKB+MMD


    #对比损失
    parser.add_argument('--constract_loss_metric', type=bool, default=False, help='whether use constract_loss metric')
    parser.add_argument('--constract_loss', type=str, choices=['constract_center_loss','SupervisedContrastiveLoss',
                                                               'constract_loss', 'CORAL','LMMD'], 
                        default='constract_loss', help='which distance loss you use')
    #隐式距离---对抗域损失
    parser.add_argument('--domain_adversarial', type=bool, default=False, help='whether use domain_adversarial')
    parser.add_argument('--adversarial_loss', type=str, choices=['DA', 'CDA', 'CDA+E'], 
                        default='b', help='which adversarial loss you use')
    
    #ADNN
    parser.add_argument('--ADNN', type=bool,
                        default=True, help='which distance loss you use')
    
    
    parser.add_argument('--hidden_size', type=int, default=1024, help='whether using the last batch')
    parser.add_argument('--trade_off_adversarial', type=str, default='Step', help='')
    parser.add_argument('--lam_adversarial', type=float, default=1, help='this is used for Cons')
    parser.add_argument('--transfer_loss_weight', type=float, default=0.5) #10
    
    parser.add_argument('--num_classes', type=int,
                        help='Number of classes', default=3)
    parser.add_argument('--batch_size', type=float,
                        help='batch size', default=128)    #64    128
    parser.add_argument('--nepoch', type=int,
                        help='Total epoch num', default=100)
    parser.add_argument('--lr', type=float, help='Learning rate', default=0.002)  #0.0002,0.001    [0.002, 0.01]    【0.02 0.01】
    parser.add_argument('--early_stop', type=int,
                        help='Early stoping number', default=185)  #35  85
    parser.add_argument('--seed', type=int,
                        help='Seed', default=2023)
    parser.add_argument('--weight', type=float,
                        help='Weight for adaptation loss', default=0.5)
    parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
    parser.add_argument('--decay', type=float,
                        help='L2 weight decay', default=5e-1)   #5e-3   5e-1
    parser.add_argument('--bottleneck', type=str2bool,
                        nargs='?', const=True, default=True) 
    parser.add_argument('--log_interval', type=int,
                        help='Log interval', default=30)
    parser.add_argument('--middle_epoch', type=int, default=0, help='max number of epoch')
    parser.add_argument('--gpu', type=str,
                        help='GPU ID', default='0')
    # model and data parameters     
    parser.add_argument('--data_name', type=str,choices=['CWRU_BJTU'], 
                        default='CWRU_BJTU', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default='D:\北京交通大学博士\实验数据\西储大学轴承数据中心网站', help='the directory of the data')
    parser.add_argument('--transfer_task', type=list, default=[[3], [1]], help='transfer learning tasks')
    parser.add_argument('--normlizetype', type=str, default='mean-std', help='nomalization type')
    parser.add_argument('--last_batch', type=bool, default=True, help='whether using the last batch')


    parser.add_argument('--gama1', type=int, help='xishu', default=1)
    parser.add_argument('--gama2', type=int, help='xishu', default=1)
    parser.add_argument('--gama3', type=int, help='xishu', default=0)
    
    parser.add_argument('--arf', type=int, help='xishu', default=2)
    parser.add_argument('--bata', type=int, help='xishu', default=1)
    # args = parser.parse_args()
    args, _ = parser.parse_known_args()
    return args


args = get_args()