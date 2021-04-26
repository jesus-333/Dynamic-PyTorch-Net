import torch
from torch import nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

import numpy as np
import math
import pickle

#%%

def getActivationList():
    """
    Method that return a list with the activation function of pytorch

    Returns
    -------
    act = python list with inside the pytorch activation function (with standar parameter)

    """
        
    #Define the activation function
    act = []
   
    act.append(nn.ReLU())               # 0
    act.append(nn.LeakyReLU())          # 1
    act.append(nn.SELU())               # 2
    act.append(nn.ELU())                # 3
    act.append(nn.GELU())               # 4
    
    act.append(nn.Sigmoid())            # 5
    act.append(nn.Tanh())               # 6
    act.append(nn.Hardtanh())           # 7
    act.append(nn.Hardshrink())         # 8
    
    act.append(nn.LogSoftmax(dim = 1))  # 9
    act.append(nn.Softmax(dim = 1))     # 10
    
    act.append(nn.Identity())           # 11
    # Linear Combination layer          # 12
    
    return act

def getPoolingList(kernel = 2, stride = 4, padding = 0, size = (1,1)):        
    tmp_pool_list = []
    # tmp_pool_list.append(nn.MaxPool2d(kernel_size = kernel, stride = stride, padding = padding))  # 0
    # tmp_pool_list.append(nn.AvgPool2d(kernel_size = kernel, stride = stride, padding = padding))  # 1
    tmp_pool_list.append(nn.MaxPool2d(kernel_size = kernel))  # 0
    tmp_pool_list.append(nn.AvgPool2d(kernel_size = kernel))  # 1
    tmp_pool_list.append(nn.AdaptiveAvgPool2d(output_size = size))  # 2    
    
    return tmp_pool_list


def getPoolingListV2(size = (1,2), stride = (1,2)):        
    tmp_pool_list = []
    # tmp_pool_list.append(nn.MaxPool2d(kernel_size = kernel, stride = stride, padding = padding))  # 0
    # tmp_pool_list.append(nn.AvgPool2d(kernel_size = kernel, stride = stride, padding = padding))  # 1
    tmp_pool_list.append(nn.MaxPool2d(kernel_size = size, stride = stride))  # 0
    tmp_pool_list.append(nn.AvgPool2d(kernel_size = size, stride = stride))  # 1
    tmp_pool_list.append(nn.AdaptiveAvgPool2d(output_size = size))  # 2    
    
    return tmp_pool_list


class LinearCombinationForMatrix(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
         
        self.linear_combination_layer = nn.Linear(c_in, c_out)
         
    def forward(self, x):
        x =  self.linear_combination_layer(x.transpose(-1, -2))
        x = x.transpose(-2, -1)
        
        return x

