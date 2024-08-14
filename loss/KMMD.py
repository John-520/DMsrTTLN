# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:59:43 2023

@author: luzy1
"""
#!/usr/bin/env python
# encoding: utf-8

import torch



def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss

    
def guassian_kernel(source, target, source_source,
                                            target_target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = Kurtosis_sigma(source,target) #

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#



def kmmd_loss(source, target, source_source,
                                            target_target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,source_source,
                                                target_target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)


def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss











import torch
import torch.nn.functional as F

def Kurtosis_sigma(x, y):
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)
    x_mean = x.mean(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)
    x_var = x.var(dim=1, keepdim=True, unbiased=False)
    y_var = y.var(dim=1, keepdim=True, unbiased=False)
    x_kurtosis = ((x - x_mean) ** 4).mean(dim=1, keepdim=True) / (x_var ** 2)
    y_kurtosis = ((y - y_mean) ** 4).mean(dim=1, keepdim=True) / (y_var ** 2)
    kurtosis_diff = torch.abs(x_kurtosis - y_kurtosis)
    sigma = kurtosis_diff.sum()

    return sigma














