# -*- coding: utf-8 -*-


import torch
from torch import nn

from jesus_support import getActivationList, getPoolingList, convOutputShape

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

class DynamicCNN(nn.Module):
    
    def __init__(self, parameters, print_var = False):
        super().__init__()
        
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Parameters recovery and check
        
        # Set device for the training/execution
        if("device" in parameters.keys()): self.device = parameters["device"]
        else: self.device = device = torch.device("cpu") 
        
        # Set the number of layers for convolutional part and linear part
        if("layers_cnn" in parameters.keys() and "layers_ff" in parameters.keys()): 
            layers_cnn = int(parameters["layers_cnn"]) #0
            layers_ff = int(parameters["layers_ff"]) #1
            
            if(print_var): print("Layer CNN:           {}\nLayer Linear:        {}".format(layers_cnn, layers_ff))
        else:
             raise Exception("No \"layers_cnn\" key or \"layers_ff\" inside the paramters dictionary") 
        
        # Set activation functions for each layer
        act = getActivationList()
        if("activation_list" in parameters.keys()): 
            activation_list = parameters["activation_list"] 
            
            # Check activation list length (N.B the +1 is added because there is the flatten layer between the cnn and the feed-forward part)
            if(len(activation_list) != layers_cnn + layers_ff + 1):raise Exception("wrong number of elements in activation_list") 
            
            # Create the activation list of the two part of the network
            activation_list_cnn = activation_list[0:layers_cnn]
            activation_list_ff = activation_list[(layers_cnn + 1):]
            activation_flatten = activation_list[layers_cnn]
            
            if(print_var): print("Activation CNN:      {}\nActivation Linear:   {}\nActivation Flatten:  {}".format(activation_list_cnn, activation_list_ff, activation_flatten))
        else: 
            raise Exception("No \"activation_list\" key inside the paramters dictionary") 
        
        # Set kernel list
        if("kernel_list" in parameters.keys()): 
            kernel_list = convertTupleElementToInt(parameters["kernel_list"])
            
            # Check kernel list length
            if(len(kernel_list) != layers_cnn):raise Exception("Wrong number of elements in kernel_list")
            
            if(print_var): print("Kernels:             {}".format(kernel_list))
        else: 
            raise Exception("No \"kernel_list\" key inside the paramters dictionary")
        
        # Set filter list
        if("filters_list" in parameters.keys()): 
            filters_list = convertTupleElementToInt(parameters["filters_list"])
            
            # Check filter list length
            if(len(filters_list) != layers_cnn):raise Exception("Wrong number of elements in filters_list") 
            
            if(print_var): print("Filters/Channels:    {}".format(filters_list))
        else: 
            raise Exception("No \"filters_list\" key inside the paramters dictionary")
        
        # Set stride list
        if("stride_list" in parameters.keys()): 
            stride_list = convertTupleElementToInt(parameters["stride_list"])
            
            # Check stride list length
            if(len(stride_list) != layers_cnn):raise Exception("Wrong number of elements in stride_list") 
            
            if(print_var): print("Stride List:         {}".format(stride_list))
        else: 
            # If no stride provided create a vector to set every stride to defualt value of conv2D
            stride_list = np.ones(layers_cnn).astype(int)
            if(print_var): print("Stride List:         {}".format(stride_list))
        
        # Set padding list
        if("padding_list" in parameters.keys()): 
            padding_list = convertTupleElementToInt(parameters["padding_list"])
            
            # Check padding list length
            if(len(padding_list) != layers_cnn):raise Exception("Wrong number of elements in padding_list") 
            
            if(print_var): print("Padding List:        {}".format(padding_list))
        else: 
            # If no padding provided create a vector to set every pad to defualt value of conv2D
            padding_list = np.zeros(layers_cnn).astype(int)
            if(print_var): print("Padding List:        {}".format(padding_list))
        
        # Set pooling list
        if("pooling_list" in parameters.keys()): 
            pooling_list = parameters["pooling_list"]
            
            # Check pooling length
            if(len(pooling_list) != layers_cnn):raise Exception("Wrong number of elements in pooling_list")
            
            if(print_var): print("Pooling List:        {}".format(pooling_list))
        else: 
            # If no pooling provided create a vector of negative number so no pool layer will be added
            pooling_list = np.ones(layers_cnn).astype(int) * -1
            if(print_var): print("Pooling List:        {}".format(pooling_list))
            
        # Set groups list
        if("groups_list" in parameters.keys()): 
            groups_list = parameters["groups_list"]
            
            # Check group length
            if(len(groups_list) != layers_cnn):raise Exception("Wrong number of elements in group_list")
            
            if(print_var): print("Groups List:         {}".format(groups_list))
        else: 
            # If no pooling provided create a vector of negative number so no pool layer will be added
            groups_list = np.ones(layers_cnn).astype(int)
            if(print_var): print("Groups List:         {}".format(groups_list))
            
        # Set Batch Normalization list
        if("CNN_normalization_list" in parameters.keys()): 
            CNN_normalization_list = parameters["CNN_normalization_list"]
            
            # Check batch_normalization_list list length
            if(len(CNN_normalization_list) != layers_cnn):raise Exception("Wrong number of elements in CNN_normalization_list")
            
            if(print_var): print("CNN Normalization:   {}".format(CNN_normalization_list))
        else: 
            # If no pooling provided create a vector of negative number so no pool layer will be added
            CNN_normalization_list = np.ones(layers_cnn).astype(int) * -1
            CNN_normalization_list = CNN_normalization_list > 100
            if(print_var): print("CNN Normalization:   {}".format(CNN_normalization_list))
            
        # Set dropout list
        if("dropout_list" in parameters.keys()): 
            dropout_list = parameters["dropout_list"]
            
            # Check pooling length
            if(len(dropout_list) != layers_cnn + layers_ff + 1):raise Exception("Wrong number of elements in dropout_list")
            
            dropout_list_cnn = dropout_list[0:layers_cnn]
            dropout_list_ff = dropout_list[(layers_cnn + 1):]
            dropout_flatten = dropout_list[layers_cnn]
            
            if(print_var): print("Dropout List:        {}".format(dropout_list))
        else: 
            # If no pooling provided create a vector of negative number so no pool layer will be added
            dropout_list = np.ones(layers_cnn + layers_ff + 1).astype(int) * -1
            
            dropout_list_cnn = dropout_list[0:layers_cnn]
            dropout_list_ff = dropout_list[(layers_cnn + 1):]
            dropout_flatten = dropout_list[layers_cnn]
            
            if(print_var): print("Dropout List:        {}".format(dropout_list))
        
        # Set neuron list
        if("neurons_list" in parameters.keys()): 
            neurons_list = parameters["neurons_list"]
            
            # Check activation list length
            if(len(neurons_list) != layers_ff):raise Exception("Wrong number of elements in neurons_list") 
            
            if(print_var): print("Neurons List:        {}\n".format(neurons_list))
        else: 
            # raise Exception("No \"Neurons_list\" key inside the paramters dictionary")
            neurons_list = []
        
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # CNN Construction
        
        # Temporary variable used to track the change in dimensions of the input
        tmp_input = torch.ones((1, filters_list[0][0], parameters["h"], parameters["w"]))
        
        # Temporay list to store the layer
        tmp_list = []

        # Construction cycle
        for kernel, n_filter, stride, padding, pool, activation, normalization, p_dropout, groups in zip(kernel_list, filters_list, stride_list, padding_list, pooling_list, activation_list_cnn, CNN_normalization_list, dropout_list_cnn, groups_list):
            
            # Create the convolutional layer and add to the list
            if(groups == 1): tmp_cnn_layer = nn.Conv2d(in_channels = int(n_filter[0]), out_channels = int(n_filter[1]), kernel_size = kernel, stride = stride, padding = padding)
            else: tmp_cnn_layer = nn.Conv2d(in_channels = int(n_filter[0]), out_channels = int(n_filter[1]), kernel_size = kernel, stride = stride, padding = padding, groups = groups)
            
            tmp_list.append(tmp_cnn_layer)
            
            # Keep track of the outupt dimension
            tmp_input = tmp_cnn_layer(tmp_input)
            
            # (OPTIONAL) add batch normalization
            if(normalization): tmp_list.append(nn.BatchNorm2d(num_features = int(n_filter[1])))
            
            # (OPTIONAL) Add the activation 
            if(activation != -1): tmp_list.append(act[activation])
            
            # (OPTIONAL) Add max pooling
            if(pool != -1):
                # Retrieve the pooling list (with a cast to int for the kernel)
                pool_kernel = (int(pool[1][0]), int(pool[1][1]))
                pool_layer_list = getPoolingList(kernel = pool_kernel)
                
                # Create the pool layer and add to the list.
                tmp_pooling_layer = pool_layer_list[pool[0]]
                tmp_list.append(tmp_pooling_layer)

                # Keep track of the output dimension
                tmp_input = tmp_pooling_layer(tmp_input)
                
            # (OPTIONAL) Dropout
            if(p_dropout > 0 and p_dropout < 1): tmp_list.append(torch.nn.Dropout(p = p_dropout))
            
        # Creation of the sequential object to store all the layer
        self.cnn = nn.Sequential(*tmp_list)
        
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Flatten layer
        
        self.flatten_neurons = tmp_input.shape[1] * tmp_input.shape[2] * tmp_input.shape[3]
        
        if(layers_ff == 0):
            if(activation_flatten != -1): self.flatten_layer = act[activation_flatten]
            else: self.flatten_layer = nn.Identity()
            
            if(print_var): print("Flatten layer:       {}".format(self.flatten_neurons))
        else:
            tmp_flatten_layer = nn.Linear(self.flatten_neurons, neurons_list[0][0])
            
            tmp_list = []
            tmp_list.append(tmp_flatten_layer)
            
            if(activation_flatten != -1): tmp_list.append(act[activation_flatten])
            if(dropout_flatten > 0 and dropout_flatten < 1): tmp_list.append(torch.nn.Dropout(p = dropout_flatten))
    
            self.flatten_layer = nn.Sequential(*tmp_list)
        
            if(print_var): print("Flatten layer:       {}".format([self.flatten_neurons, neurons_list[0][0]]))
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Feed-Forward (Linear) construction
        
        # Temporay list to store the layer
        tmp_list = []
        
        # Construction cycle
        for neurons, activation, p_dropout in zip(neurons_list, activation_list_ff, dropout_list_ff):
            tmp_linear_layer = nn.Linear(neurons[0], neurons[1])
            tmp_list.append(tmp_linear_layer)
            
            # (OPTIONAL) Add the activation 
            if(activation != -1): tmp_list.append(act[activation])
            
            # (OPTIONAL) Dropout
            if(p_dropout > 0 and p_dropout < 1): tmp_list.append(torch.nn.Dropout(p = p_dropout))
        
        # Creation of the sequential object to store all the layer
        self.ff = nn.Sequential(*tmp_list)
        
        
    def forward(self, x):
        # Convolutional section
        x = self.cnn(x)
        
        # Flatten layer
        x = x.view([x.size(0), -1])
        x = self.flatten_layer(x)
        
        # Feed-forward (linear) section
        x = self.ff(x)
        
        return x

        
#%%

def convertArrayInTupleList(array):
    """
    Convert an array (or a list) of element in a list of tuple where each element is a tuple with two sequential element of the original array/list

    Parameters
    ----------
    array : numpy array/list

    Returns
    -------
    tuple_list. List of tuple
        Given the input array = [a, b, c, d ...] the tuple_list will be [(a, b), (b, c), (c, d) ...]

    """
    
    tuple_list = []
    
    for i in range(len(array) - 1):
        tmp_tuple = (array[i], array[i + 1])
        
        tuple_list.append(tmp_tuple)
        
    return tuple_list

def convertTupleElementToInt(tuple_list):
    """
    Convert a list of tuple in the same list of tuple but with tuple elements cast to int
    N.B. The tuples must contain two elements

    """
    
    tuple_int_list = []
    
    for tup in tuple_list:
        
        tmp_tuple = (int(tup[0]), int(tup[1]))
        
        tuple_int_list.append(tmp_tuple)
        
    return tuple_int_list

