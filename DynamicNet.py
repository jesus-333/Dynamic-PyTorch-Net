# -*- coding: utf-8 -*-


import torch
from torch import nn

from support_DynamicNet import getActivationList, getPoolingList, convOutputShape

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

class DynamicCNN(nn.Module):
    
    def __init__(self, parameters, print_var = False, tracking_input_dimension = False):
        super().__init__()
        
        self.print_var = print_var
        self.tracking_input_dimension = tracking_input_dimension
        
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Parameters recovery and check
        
        # Set device for the training/execution
        if("device" in parameters.keys()): self.device = parameters["device"]
        else: self.device = device = torch.device("cpu") 
        
        # Set the number of layers for convolutional part
        if("layers_cnn" in parameters.keys()): 
            layers_cnn = int(parameters["layers_cnn"]) #0
    
            if(print_var): print("Layer CNN:           {}".format(layers_cnn))
        else:
            layers_cnn = 0;
            if(print_var): print("Layer CNN:           {}".format(layers_cnn))
             # raise Exception("No \"layers_cnn\" key inside the paramters dictionary") 
        
        # Set the number of layers for linear part
        if("layers_ff" in parameters.keys()): 
            layers_ff = int(parameters["layers_ff"]) #1
            
            if(print_var): print("Layer Linear:        {}".format(layers_ff))
        else:
            layers_ff = 0
            if(print_var): print("Layer Linear:        {}".format(layers_ff))
            # raise Exception("No \"layers_ff\" key inside the paramters dictionary") 
        
        if(layers_cnn == 0 and layers_ff == 0): raise Exception("Both  \"layers_cnn\" and \"layers_ff\" are set to 0. You must have at least one layer.") 
        self.layers_cnn, self.layers_ff = layers_cnn, layers_ff
        
        # Set activation functions for each layer
        act = getActivationList()
        if("activation_list" in parameters.keys()): 
            activation_list = parameters["activation_list"] 
            
            # Check activation list length (N.B the +1 is added because there is the flatten layer between the cnn and the feed-forward part)
            if(len(activation_list) != layers_cnn + layers_ff + 1): raise Exception("wrong number of elements in activation_list") 
            
            # Create the activation list of the two part of the network
            activation_list_cnn = activation_list[0:layers_cnn]
            activation_list_ff = activation_list[(layers_cnn + 1):]
            activation_flatten = activation_list[layers_cnn]
            
            if(print_var): print("Activation CNN:      {}\nActivation Linear:   {}\nActivation Flatten:  {}".format(activation_list_cnn, activation_list_ff, activation_flatten))
        else: 
            raise Exception("No \"activation_list\" key inside the paramters dictionary") 
        
        if(layers_cnn != 0):
            # Set kernel list
            if("kernel_list" in parameters.keys() and layers_cnn != 0): 
                kernel_list = convertTupleElementToInt(parameters["kernel_list"])
                
                # Check kernel list length
                if(len(kernel_list) != layers_cnn): raise Exception("Wrong number of elements in kernel_list")
                
                if(print_var): print("Kernels:             {}".format(kernel_list))
            else: 
                if(print_var): print("Kernels:             {}".format(kernel_list))
                # raise Exception("No \"kernel_list\" key inside the paramters dictionary")
            
            # Set filter list
            if("filters_list" in parameters.keys() and layers_cnn != 0): 
                filters_list = convertTupleElementToInt(parameters["filters_list"])
                
                # Check filter list length
                if(len(filters_list) != layers_cnn): raise Exception("Wrong number of elements in filters_list") 
                
                if(print_var): print("Filters/Channels:    {}".format(filters_list))
            else: 
                raise Exception("No \"filters_list\" key inside the paramters dictionary")
            
            # Set stride list
            if("stride_list" in parameters.keys() and layers_cnn != 0): 
                stride_list = convertTupleElementToInt(parameters["stride_list"])
                
                # Check stride list length
                if(len(stride_list) != layers_cnn): raise Exception("Wrong number of elements in stride_list") 
                
                if(print_var): print("Stride List:         {}".format(stride_list))
            else: 
                # If no stride provided create a vector to set every stride to defualt value of conv2D
                stride_list = np.ones(layers_cnn).astype(int)
                if(print_var): print("Stride List:         {}".format(stride_list))
            
            # Set padding list
            if("padding_list" in parameters.keys() and layers_cnn != 0): 
                padding_list = convertTupleElementToInt(parameters["padding_list"])
                
                # Check padding list length
                if(len(padding_list) != layers_cnn): raise Exception("Wrong number of elements in padding_list") 
                
                if(print_var): print("Padding List:        {}".format(padding_list))
            else: 
                # If no padding provided create a vector to set every pad to defualt value of conv2D
                padding_list = np.zeros(layers_cnn).astype(int)
                if(print_var): print("Padding List:        {}".format(padding_list))
            
            # Set pooling list
            if("pooling_list" in parameters.keys() and layers_cnn != 0): 
                pooling_list = parameters["pooling_list"]
                
                # Check pooling length
                if(len(pooling_list) != layers_cnn): raise Exception("Wrong number of elements in pooling_list")
                
                if(print_var): print("Pooling List:        {}".format(pooling_list))
            else: 
                # If no pooling provided create a vector of negative number so no pool layer will be added
                pooling_list = np.ones(layers_cnn).astype(int) * -1
                if(print_var): print("Pooling List:        {}".format(pooling_list))
                
            # Set groups list
            if("groups_list" in parameters.keys() and layers_cnn != 0): 
                groups_list = parameters["groups_list"]
                
                # Check group length
                if(len(groups_list) != layers_cnn): raise Exception("Wrong number of elements in group_list")
                
                if(print_var): print("Groups List:         {}".format(groups_list))
            else: 
                # If no groups provided create a vector of ones number so hte group will be set to its default value of 1
                groups_list = np.ones(layers_cnn).astype(int)
                if(print_var): print("Groups List:         {}".format(groups_list))
                
            # Set Batch Normalization list
            if("CNN_normalization_list" in parameters.keys() and layers_cnn != 0): 
                CNN_normalization_list = parameters["CNN_normalization_list"]
                
                # Check batch_normalization_list list length
                if(len(CNN_normalization_list) != layers_cnn): raise Exception("Wrong number of elements in CNN_normalization_list")
                
                if(print_var): print("CNN Normalization:   {}".format(CNN_normalization_list))
            else: 
                # If no Batch was provided create a vector of negative number so no Batch layer will be added
                CNN_normalization_list = np.ones(layers_cnn).astype(int) * -1
                CNN_normalization_list = CNN_normalization_list > 100
                if(print_var): print("CNN Normalization:   {}".format(CNN_normalization_list))
            
        # Set dropout list
        if("dropout_list" in parameters.keys()): 
            dropout_list = parameters["dropout_list"]
            
            # Check dropout list length
            if(len(dropout_list) != layers_cnn + layers_ff + 1): raise Exception("Wrong number of elements in dropout_list")
            
            dropout_list_cnn = dropout_list[0:layers_cnn]
            dropout_list_ff = dropout_list[(layers_cnn + 1):]
            dropout_flatten = dropout_list[layers_cnn]
            
            if(print_var): print("Dropout List:        {}".format(dropout_list))
        else: 
            # If no dropout was provided create a vector of negative number so no dropout layer will be added
            dropout_list = np.ones(layers_cnn + layers_ff + 1).astype(int) * -1
            
            dropout_list_cnn = dropout_list[0:layers_cnn]
            dropout_list_ff = dropout_list[(layers_cnn + 1):]
            dropout_flatten = dropout_list[layers_cnn]
            
            if(print_var): print("Dropout List:        {}".format(dropout_list))
            
        # Set bias list
        if("bias_list" in parameters.keys()): 
            bias_list = parameters["bias_list"]
            
            # Check bias list length
            if(len(bias_list) != layers_cnn + layers_ff + 1): raise Exception("Wrong number of elements in bias_list")
            
            bias_list_cnn = bias_list[0:layers_cnn]
            bias_list_ff = bias_list[(layers_cnn + 1):]
            bias_list_flatten = bias_list[layers_cnn]
            
            if(print_var): print("Bias List:           {}".format(bias_list))
        else: 
            # If no bias was provided create a vector of negative number so no bias will be added
            bias_list = np.ones(layers_cnn + layers_ff + 1).astype(int) * -1
            bias_list = bias_list < 1000
            
            bias_list_cnn = bias_list[0:layers_cnn]
            bias_list_ff = bias_list[(layers_cnn + 1):]
            bias_list_flatten = bias_list[layers_cnn]
            
            if(print_var): print("Bias List:           {}".format(bias_list))
        
        # Set neuron list
        if("neurons_list" in parameters.keys()): 
            neurons_list = parameters["neurons_list"]
            
            # Check activation list length
            if(len(neurons_list) != layers_ff): raise Exception("Wrong number of elements in neurons_list") 
            
            neurons_list = convertArrayInTupleList(neurons_list)
            
            if(print_var): print("Neurons List:        {}".format(neurons_list))
        else: 
            # raise Exception("No \"Neurons_list\" key inside the paramters dictionary")
            neurons_list = []
            if(print_var): print("Neurons List:        {}".format(neurons_list))
        
        # Add a empty line
        if(print_var): print()
        
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # CNN Construction
        
        # Temporary variable used to track the change in dimensions of the input
        if(layers_cnn != 0):
            tmp_input = torch.ones((1, filters_list[0][0], parameters["h"], parameters["w"]))
            if(tracking_input_dimension): 
                print("# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # ")
                print(tmp_input.shape, "\n")
            
            # Temporay list to store the layer
            tmp_list = []
    
            # Construction cycle
            for kernel, n_filter, stride, padding, pool, activation, normalization, p_dropout, groups, bias in zip(kernel_list, filters_list, stride_list, padding_list, pooling_list, activation_list_cnn, CNN_normalization_list, dropout_list_cnn, groups_list, bias_list_cnn):
                
                # Create the convolutional layer and add to the list
                if(groups == 1): tmp_cnn_layer = nn.Conv2d(in_channels = int(n_filter[0]), out_channels = int(n_filter[1]), kernel_size = kernel, stride = stride, padding = padding, bias = bias)
                else: tmp_cnn_layer = nn.Conv2d(in_channels = int(n_filter[0]), out_channels = int(n_filter[1]), kernel_size = kernel, stride = stride, padding = padding, groups = groups, bias = bias)
                
                tmp_list.append(tmp_cnn_layer)
                
                # Keep track of the outupt dimension
                tmp_input = tmp_cnn_layer(tmp_input)
                
                # Print the input dimensions at this step (if tracking_input_dimension is True)
                if(tracking_input_dimension): 
                    print(tmp_cnn_layer)
                    print(tmp_input.shape, "\n")
                
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
                    
                    # Print the input dimensions at this step (if tracking_input_dimension is True)
                    if(tracking_input_dimension): 
                        print(tmp_pooling_layer)
                        print(tmp_input.shape)
                    
                # (OPTIONAL) Dropout
                if(p_dropout > 0 and p_dropout < 1): tmp_list.append(torch.nn.Dropout(p = p_dropout))
                
            # Creation of the sequential object to store all the layer
            self.cnn = nn.Sequential(*tmp_list)
            
            # Plot a separator
            if(tracking_input_dimension): print("# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #\n")
        
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            # Flatten layer
            
            self.flatten_neurons = tmp_input.shape[1] * tmp_input.shape[2] * tmp_input.shape[3]
            
            if(layers_ff == 0):
                if(activation_flatten != -1): self.flatten_layer = act[activation_flatten]
                else: self.flatten_layer = nn.Identity()
                
                if(print_var): print("Flatten layer:       {}\n".format(self.flatten_neurons))
            else:
                if(layers_ff == 1): tmp_flatten_layer = nn.Linear(self.flatten_neurons, neurons_list[0], bias = bias_list_flatten)
                else: tmp_flatten_layer = nn.Linear(self.flatten_neurons, neurons_list[0][0], bias = bias_list_flatten)
                
                tmp_list = []
                tmp_list.append(tmp_flatten_layer)
                
                if(activation_flatten != -1): tmp_list.append(act[activation_flatten])
                if(dropout_flatten > 0 and dropout_flatten < 1): tmp_list.append(torch.nn.Dropout(p = dropout_flatten))
        
                self.flatten_layer = nn.Sequential(*tmp_list)
            
                if(print_var): 
                    if(layers_ff == 1): print("Flatten layer:       {}\n".format([self.flatten_neurons, neurons_list[0]]))
                    else: print("Flatten layer:       {}\n".format([self.flatten_neurons, neurons_list[0][0]]))
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Feed-Forward (Linear) construction
        
        if(layers_ff > 1):
            # Temporay list to store the layer
            tmp_list = []
            
            # Construction cycle
            for neurons, activation, p_dropout, bias in zip(neurons_list, activation_list_ff, dropout_list_ff, bias_list_ff):
                if(layers_ff == 1 and layers_cnn == 0): # Case for a single layer feed-forward network (perceptron style)
                    tmp_linear_layer = nn.Linear(parameters["h"] * parameters["w"], neurons, bias = bias)
                else:
                    tmp_linear_layer = nn.Linear(neurons[0], neurons[1], bias = bias)
                tmp_list.append(tmp_linear_layer)
                
                # (OPTIONAL) Add the activation 
                if(activation != -1): tmp_list.append(act[activation])
                
                # (OPTIONAL) Dropout
                if(p_dropout > 0 and p_dropout < 1): tmp_list.append(torch.nn.Dropout(p = p_dropout))
            
            # Creation of the sequential object to store all the layer
            self.ff = nn.Sequential(*tmp_list)
        else: self.ff = []
        
        
    def forward(self, x):
        if(self.layers_cnn != 0):
            # Convolutional section
            x = self.cnn(x)
            
            # Flatten layer
            x = x.view([x.size(0), -1])
            x = self.flatten_layer(x)
        
        # Feed-forward (linear) section
        if(len(self.ff) > 0): x = self.ff(x)
        
        return x
    
    
    def printNetwork(self, separator = False):
        depth = 0
        
        # Iterate through the module of the network
        for name, module in self.named_modules():
            
            # Iterate through the sequential block
            # Since in the iteration the sequential blocks and the modules inside the sequential block appear twice I only take the sequenial block
            if(type(module) == torch.nn.modules.container.Sequential):
                for layer in module:
                    
                    # Print layer
                    print("DEPTH:", depth, "\t- ", layer)
                    
                    # Incrase depth
                    depth += 1
                
                if(separator): print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - \n")
                
                if(name == 'cnn'):
                    # Add reshape "layer"
                    print("DEPTH:", depth, "\t- ", "x.view([x.size(0), -1])")
                    if(separator): print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - \n")
                    depth += 1
    
    
    def getMiddleResults(self, x, input_depth, ignore_dropout = True):
        actual_depth = 0
        
        # Iterate through the module of the network
        for name, module in self.named_modules():
            
            # Iterate through the sequential block
            # Since in the iteration the sequential blocks and the modules inside the sequential block appear twice I only take the sequenial block
            if(type(module) == torch.nn.modules.container.Sequential):
                for layer in module:
                    # Evaluate the value of the input at this level
                    x = layer(x)
                    
                    # If I reach the desire level I stop
                    if(actual_depth == input_depth): return x
                    
                    # Increase depth level
                    actual_depth += 1
                
                # Reshape after the CNN block
                if(name == 'cnn'): 
                    x = x.view([x.size(0), -1])
                    if(actual_depth == input_depth): return x
                    actual_depth += 1
        
        # If this istruction is reached it means that the input flow inside all the network. 
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

