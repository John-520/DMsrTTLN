# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 17:48:10 2021

@author: 29792
"""
import torch
def Shuffle(x, y,random=None, int=int):
       """x, random=random.random -> shuffle list x in place; return None.
       Optional arg random is a 0-argument function returning a random
       float in [0.0, 1.0); by default, the standard random.random.
       """
       if random is None:
           random = random #random=random.random
       #转成numpy
       if torch.is_tensor(x)==True:
             x=x.numpy()
       if torch.is_tensor(y) == True:
             y=y.numpy()
       #开始随机置换
       for i in range(len(x)):
           j = int(random() * (i + 1))
           if j<=len(x)-1:#交换
               x[i],x[j]=x[j],x[i]
               y[i],y[j]=y[j],y[i]
       #转回tensor
       x=torch.from_numpy(x)
       y=torch.from_numpy(y)
       return x,y
