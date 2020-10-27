import torch
from torch import nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

import numpy as np
import matplotlib.pyplot as plt
import random
import math
import pickle

def getActivationList():
    """
    Method that return a list with the activation function of pytorch

    Returns
    -------
    act = python list with inside the pytorch activation function (with standar parameter)

    """
        
    #Define the activation function
    act = []
    act.append(nn.Sigmoid())    # 0
    act.append(nn.ReLU())       # 1
    act.append(nn.LeakyReLU())  # 2
    act.append(nn.Tanh())       # 3
    act.append(nn.SELU())       # 4
    act.append(nn.Hardshrink()) # 5
    act.append(nn.Hardtanh())   # 6
    act.append(nn.LogSoftmax)   # 7
    act.append(nn.ELU())        # 8
    act.append(nn.Softmax())    # 9
    
    return act

def getPoolingList(kernel = 2, stride = 4, padding = 0):
    if(padding == 0 and type(kernel) is tuple): 
        x1 = int(kernel[0]/2)  
        x2 = int(kernel[1]/2)  
        padding = (x1, x2)

        
    tmp_pool_list = []
    # tmp_pool_list.append(nn.MaxPool2d(kernel_size = kernel, stride = stride, padding = padding))  # 0
    # tmp_pool_list.append(nn.AvgPool2d(kernel_size = kernel, stride = stride, padding = padding))  # 1
    tmp_pool_list.append(nn.MaxPool2d(kernel_size = kernel))  # 0
    tmp_pool_list.append(nn.AvgPool2d(kernel_size = kernel))  # 1
    
    return tmp_pool_list


    
def KFoldIndexing(n_entry, k):
    """
    Return a list of index for the k-folding. (This function was write to be reused)

    Parameters
    ----------
    n_entry : int
        Total number of entru(sample) in the database.
    k : int
        Number of fold in which divide the database.

    Returns
    -------
    list_of_index : list
        List of tuple. Each tuple contain two entry. The i-th tuple contain the index for the test set of the i-th iteration of the k-fold.

    """
    list_of_index = []
    step = int(n_entry / k)
    
    for i in range(k): list_of_index.append((i * step, (i+1) * step))
    
    return list_of_index

def KFoldOnDataset(dataset, k, batch_size = 100):
    list_test_index = KFoldIndexing(len(dataset), k)
    
    fold_list = []
    index_list = list(range(len(dataset)))
    
    for index, i in zip(list_test_index, range(len(list_test_index))):
        if 0 in index:
            # First couple of index
            dataloader_validation = DataLoader(Subset(dataset, index_list[index[0]:index[1]]), batch_size = batch_size, shuffle = True);
            dataloader_train = DataLoader(Subset(dataset, index_list[index[1]:]), batch_size = batch_size, shuffle = True);
        
            fold_list.append((dataloader_train, dataloader_validation))
        elif i == (len(list_test_index) - 1):
            # Last couple of index
            dataloader_validation = DataLoader(Subset(dataset, index_list[index[0]:]), batch_size = batch_size, shuffle = True);
            dataloader_train = DataLoader(Subset(dataset, index_list[0:index[0]]), batch_size = batch_size, shuffle = True);
        
            fold_list.append((dataloader_train, dataloader_validation))
        else:
            # All others couple of index
            dataloader_validation = DataLoader(Subset(dataset, index_list[index[0]:index[1]]), batch_size = batch_size, shuffle = True);
            
            tmp_list = []
            tmp_list.append(Subset(dataset, index_list[0:index[0]]))
            tmp_list.append(Subset(dataset, index_list[index[1]:]))
            dataloader_train = DataLoader(data.ConcatDataset(tmp_list), batch_size = batch_size, shuffle = True);
                                                                
            fold_list.append((dataloader_train, dataloader_validation))
    
    return fold_list



def convOutputShape(h_w, kernel_size = 1, stride = 1, pad = 0, dilation = 1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = math.floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = math.floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return [h, w]


def save_data(data, file_name):
    with open(file_name + ".dat", "wb") as f:
        pickle.dump(data, f)
        
def load_data(file_name):
    with open(file_name + ".dat") as f:
            x = pickle.load(f, encoding="bytes")
        
    print(x) 
    return x

