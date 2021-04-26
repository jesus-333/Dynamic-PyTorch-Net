# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.onnx

from support_DynamicNet import getActivationList, getPoolingList, LinearCombinationForMatrix

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from scipy.io import savemat, loadmat

import os

#%%

class DynamicCNN(nn.Module):
    
    def __init__(self, parameters, print_var = False, tracking_input_dimension = False):
        super().__init__()
        
        self.print_var = print_var
        self.tracking_input_dimension = tracking_input_dimension
        self.count_block = 0
        self.parameters_creation = parameters
        
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Parameters recovery and check
        
        # Set device for the training/execution
        if("device" in parameters.keys()): self.device = parameters["device"]
        else: self.device = device = torch.device("cpu") 
        
        # Set if activate or not the multi block structure
        if("multi_block_structure" in parameters.keys()): self.multi_block_structure = parameters["multi_block_structure"]
        else: self.multi_block_structure = False 
        
        # Set the number of layers for convolutional part 
        if("layers_cnn" in parameters.keys()): 
            layers_cnn = int(parameters["layers_cnn"])
            self.layers_cnn = layers_cnn
            
            if(print_var): print("Layer CNN:           {}".format(layers_cnn))
        else:
             raise Exception("No \"layers_cnn\" key inside the paramters dictionary")
        
        # Set the number of layers for linear part
        if("layers_ff" in parameters.keys()): 
            layers_ff = int(parameters["layers_ff"])
            self.layers_ff = layers_ff
            
            if(print_var): print("Layer Linear:        {}".format(layers_ff))
        else:
             raise Exception("No \"layers_ff\" inside the paramters dictionary")
          
        if(layers_cnn + layers_ff == 0): raise Exception("\"layers_cnn\" + \"layers_ff\" = 0. The network must have at least 1 layer.")
          
        # Set if add a flatten layer
        if("add_flatten_layer" in parameters.keys()): self.add_flatten_layer = parameters["add_flatten_layer"] 
        else: 
            # If not specified follow one of the 3 cases
            # (1) both cnn and ff section are created. So flatten layer is needed
            # (2) only ff section is created (layers_cnn == 0) so the flatten layer isn't needed
            # (3) default (technically useless)
            if(layers_cnn > 0 and layers_ff > 0): self.add_flatten_layer = True
            elif(layers_cnn == 0): self.add_flatten_layer = False
            else: self.add_flatten_layer = True
        
        # Set activation functions for each layer
        act = getActivationList()
        if("activation_list" in parameters.keys()): 
            activation_list = parameters["activation_list"] 
            
            # Check activation list length (N.B the +1 is added because there is the flatten layer between the cnn and the feed-forward part)
            if(self.add_flatten_layer): 
                # Default cases. Both cnn AND ff are created.
                if(len(activation_list) != layers_cnn + layers_ff + 1): raise Exception("wrong number of elements in activation_list")
            else: 
                # Particular case. Only cnn OR ff is created.
                if(len(activation_list) != layers_cnn + layers_ff): raise Exception("wrong number of elements in activation_list")
             
            # Create the activation list of the two part of the network
            activation_list_cnn = activation_list[0:layers_cnn]
            if(self.add_flatten_layer):
                activation_list_ff = activation_list[(layers_cnn + 1):]
                activation_flatten = activation_list[layers_cnn]
            else: 
                activation_list_ff = activation_list[layers_cnn:] 
                activation_flatten = []
            
            if(print_var): print("Activation CNN:      {}\nActivation Linear:   {}\nActivation Flatten:  {}".format(activation_list_cnn, activation_list_ff, activation_flatten))
        else: 
            raise Exception("No \"activation_list\" key inside the paramters dictionary") 
        
        # Set kernel list
        if("kernel_list" in parameters.keys()): 
            kernel_list = convertTupleElementToInt(parameters["kernel_list"])
            
            # Check kernel list length
            if(len(kernel_list) != layers_cnn): raise Exception("Wrong number of elements in kernel_list")
            
            if(print_var): print("Kernels:             {}".format(kernel_list))
        else: 
            raise Exception("No \"kernel_list\" key inside the paramters dictionary")
        
        # Set filter list
        if("filters_list" in parameters.keys()): 
            filters_list = convertTupleElementToInt(parameters["filters_list"])
            
            # Check filter list length
            if(len(filters_list) != layers_cnn): raise Exception("Wrong number of elements in filters_list") 
            
            if(print_var): print("Filters/Channels:    {}".format(filters_list))
        else: 
            raise Exception("No \"filters_list\" key inside the paramters dictionary")
        
        # Set stride list
        if("stride_list" in parameters.keys()): 
            stride_list = convertTupleElementToInt(parameters["stride_list"])
            
            # Check stride list length
            if(len(stride_list) != layers_cnn): raise Exception("Wrong number of elements in stride_list") 
            
            if(print_var): print("Stride List:         {}".format(stride_list))
        else: 
            # If no stride provided create a vector to set every stride to defualt value of conv2D
            stride_list = np.ones(layers_cnn).astype(int)
            if(print_var): print("Stride List:         {}".format(stride_list))
        
        # Set padding list
        if("padding_list" in parameters.keys()): 
            padding_list = convertTupleElementToInt(parameters["padding_list"])
            
            # Check padding list length
            if(len(padding_list) != layers_cnn): raise Exception("Wrong number of elements in padding_list") 
            
            if(print_var): print("Padding List:        {}".format(padding_list))
        else: 
            # If no padding provided create a vector to set every pad to defualt value of conv2D
            padding_list = np.zeros(layers_cnn).astype(int)
            if(print_var): print("Padding List:        {}".format(padding_list))
        
        # Set pooling list
        if("pooling_list" in parameters.keys()): 
            pooling_list = parameters["pooling_list"]
            
            # Check pooling length
            if(len(pooling_list) != layers_cnn): raise Exception("Wrong number of elements in pooling_list")
            
            if(print_var): print("Pooling List:        {}".format(pooling_list))
        else: 
            # If no pooling provided create a vector of negative number so no pool layer will be added
            pooling_list = np.ones(layers_cnn).astype(int) * -1
            if(print_var): print("Pooling List:        {}".format(pooling_list))
            
        # Set groups list
        if("groups_list" in parameters.keys()): 
            groups_list = parameters["groups_list"]
            
            # Check group length
            if(len(groups_list) != layers_cnn): raise Exception("Wrong number of elements in group_list")
            
            if(print_var): print("Groups List:         {}".format(groups_list))
        else: 
            groups_list = np.ones(layers_cnn).astype(int)
            if(print_var): print("Groups List:         {}".format(groups_list))
            
        # Set Batch Normalization list
        if("CNN_normalization_list" in parameters.keys()): 
            CNN_normalization_list = parameters["CNN_normalization_list"]
            
            # Check batch_normalization_list list length
            if(len(CNN_normalization_list) != layers_cnn): raise Exception("Wrong number of elements in CNN_normalization_list")
            
            if(print_var): print("CNN Normalization:   {}".format(CNN_normalization_list))
        else: 
            # If no pooling provided create a vector of negative number so no normalization layer will be added
            CNN_normalization_list = np.ones(layers_cnn).astype(int) * -1
            CNN_normalization_list = CNN_normalization_list > 100
            if(print_var): print("CNN Normalization:   {}".format(CNN_normalization_list))
            
        # Set dropout list
        if("dropout_list" in parameters.keys()): 
            dropout_list = parameters["dropout_list"]
            
            # Check dropout list length
            if(self.add_flatten_layer): 
                # Default cases. Both cnn AND ff are created.
                if(len(dropout_list) != layers_cnn + layers_ff + 1): raise Exception("wrong number of elements in dropout_list")
            else: 
                # Particular case. Only cnn OR ff is created.
                if(len(dropout_list) != layers_cnn + layers_ff): raise Exception("wrong number of elements in dropout_list")
            
            dropout_list_cnn = dropout_list[0:layers_cnn]
            if(self.add_flatten_layer): 
                dropout_list_ff = dropout_list[(layers_cnn + 1):]
                dropout_flatten = dropout_list[layers_cnn]
            else: dropout_list_ff = dropout_list[layers_cnn:]
            
            if(print_var): print("Dropout List:        {}".format(dropout_list))
        else: 
            # If no pooling provided create a vector of negative number so no dropout layer will be added     
            dropout_list = np.ones(layers_cnn + layers_ff + 1) * -1
            dropout_list_cnn = dropout_list[0:layers_cnn]
            if(self.add_flatten_layer): 
                dropout_list_ff = dropout_list[(layers_cnn + 1):]
                dropout_flatten = dropout_list[layers_cnn]
            else: dropout_list_ff = np.ones(layers_cnn + layers_ff) * -1
            
            if(print_var): print("Dropout List:        {}".format(dropout_list))
            
        # Set bias list
        if("bias_list" in parameters.keys()): 
            bias_list = parameters["bias_list"]
            
            # Check bias list length
            if(self.add_flatten_layer): 
                # Default cases. Both cnn AND ff are created.
                if(len(bias_list) != layers_cnn + layers_ff + 1): raise Exception("wrong number of elements in bias_list")
            else: 
                # Particular case. Only cnn OR ff is created.
                if(len(bias_list) != layers_cnn + layers_ff): raise Exception("wrong number of elements in bias_list")
            
            bias_list_cnn = bias_list[0:layers_cnn]
            if(self.add_flatten_layer): 
                bias_list_ff = bias_list[(layers_cnn + 1):]
                bias_list_flatten = bias_list[layers_cnn]
            else: bias_list_ff = bias_list[layers_cnn:]
            
            if(print_var): print("Bias List:           {}".format(bias_list))
        else: 
            # If no bias list was provided create a vector of positive number
            bias_list = np.ones(layers_cnn + layers_ff + 1).astype(int)
            bias_list = bias_list < 1000
            
            bias_list_cnn = bias_list[0:layers_cnn]
            bias_list_ff = bias_list[(layers_cnn + 1):]
            bias_list_flatten = bias_list[layers_cnn]
            
            if(print_var): print("Bias List:           {}".format(bias_list))
        
        # Set neuron list
        if("neurons_list" in parameters.keys()): 
            neurons_list = parameters["neurons_list"]
            
            if(len(neurons_list) != layers_ff): raise Exception("Wrong number of elements in neurons_list") 
            
            if(layers_ff != 1): neurons_list = convertArrayInTupleList(neurons_list)
            
            if(print_var): print("Neurons List:        {}".format(neurons_list))
        else: 
            # raise Exception("No \"Neurons_list\" key inside the paramters dictionary")
            neurons_list = []
            if(print_var): print("Neurons List:        {}".format(neurons_list))
        
        # Add a empty line
        if(print_var): print()
        
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # CNN Construction
        if(layers_cnn > 0):
        
            # Temporary variable used to track the change in dimensions of the input
            tmp_input = torch.ones((1, filters_list[0][0], parameters["h"], parameters["w"]))
            if(tracking_input_dimension): 
                print("# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # ")
                print(tmp_input.shape, "\n")
            
            # Temporay list to store the layer
            tmp_list = []
            
            # Used only if the network is constructed with the block structure
            tmp_block_list = []
    
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
                    if(pool != -1): print(tmp_input.shape)
                    else: print(tmp_input.shape, "\n")
                
                # (OPTIONAL) add batch normalization
                if(normalization): tmp_list.append(nn.BatchNorm2d(num_features = int(n_filter[1])))
                
                # (OPTIONAL) Add the activation 
                if(activation != -1 and  activation != 12): tmp_list.append(act[activation])
                elif(activation == 12): tmp_list.append(LinearCombinationForMatrix(parameters["h"], parameters["h"]))
                
                # (OPTIONAL) Add pooling
                if(pool != -1):
                    # Retrieve the pooling list (with a cast to int for the kernel)
                    pool_kernel = (int(pool[1][0]), int(pool[1][1]))
                    if(pool[0] == 2): pool_layer_list = getPoolingList(size = pool_kernel)
                    else: pool_layer_list = getPoolingList(kernel = pool_kernel)
                    
                    # Create the pool layer and add to the list.
                    tmp_pooling_layer = pool_layer_list[pool[0]]
                    tmp_list.append(tmp_pooling_layer)
    
                    # Keep track of the output dimension
                    tmp_input = tmp_pooling_layer(tmp_input)
                    
                    # Print the input dimensions at this step (if tracking_input_dimension is True)
                    if(tracking_input_dimension): 
                        print(tmp_pooling_layer)
                        print(tmp_input.shape, "\n")
                    
                # (OPTIONAL) Dropout
                if(p_dropout > 0 and p_dropout < 1): 
                    tmp_list.append(torch.nn.Dropout(p = p_dropout))
                    # tmp_list.append(torch.nn.AlphaDropout(p = p_dropout))
                
                if(self.multi_block_structure): 
                    tmp_cnn_block = nn.Sequential(*tmp_list)
                    tmp_block_list.append(tmp_cnn_block)
                    tmp_list = []
                    self.count_block += 1
                
            # Creation of the sequential object to store all the layer
            if(self.multi_block_structure):
                self.cnn = nn.Sequential(*tmp_block_list)
            else:
                self.cnn = nn.Sequential(*tmp_list)
            
            # Plot a separator
            if(tracking_input_dimension): print("# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #\n")
        
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Flatten layer
        
        if(self.add_flatten_layer):
            self.flatten_neurons = tmp_input.shape[1] * tmp_input.shape[2] * tmp_input.shape[3]
            
            if(layers_ff == 0):
                if(activation_flatten != -1): self.flatten_layer = act[activation_flatten]
                else: self.flatten_layer = nn.Identity()
                
                if(print_var): print("Flatten layer:       {}\n".format(self.flatten_neurons))
            else:
                if(layers_ff == 1): tmp_flatten_layer = nn.Linear(self.flatten_neurons, neurons_list[0], bias = bias_list_flatten)
                else: 
                    tmp_flatten_layer = nn.Linear(self.flatten_neurons, neurons_list[0][0], bias = bias_list_flatten)
                
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
        # Convolutional section
        if(self.layers_cnn > 0):
            x = self.cnn(x)
        
        # Flatten layer
        if(self.add_flatten_layer):
            x = x.view([x.size(0), -1])
            x = self.flatten_layer(x)
        
        # Feed-forward (linear) section
        if(len(self.ff) > 0): x = self.ff(x)
        
        return x
    
    
    def printNetwork(self, separator = False):
        if(self.multi_block_structure):
            self.printNetworkMultiBlock(separator)
        else:
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
                        # N.B. Since the network is cycled in different way respect the multiblock in this case I encounter the flatten layer while I watch the cnn
                        print("DEPTH:", depth, "\t- ", "x.view([x.size(0), -1])")
                        if(separator): print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - \n")
                        depth += 1
            
                        
    def printNetworkMultiBlock(self, separator = True):
        depth = 0
        
        for name, module in self.named_modules():
            if(name != '' and name != 'cnn' and name != 'flatten_layer' and name != 'ff'):
                # print(name, hasattr(module, '__len__'))
                if not (hasattr(module, '__len__')):
                    # Print layer
                    print("DEPTH:", depth, "\t- ", module)
                        
                    # Incrase depth
                    depth += 1
                    
                    if(separator): print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - \n")
            
            if(name == 'flatten_layer'):
                # Add reshape "layer"
                print("DEPTH:", depth, "\t- ", "x.view([x.size(0), -1])")
                if(separator): print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - \n")
                depth += 1
                
    def getDepth(self):
        if(self.multi_block_structure):
            return self.getDepthMultiBlock()
        else:
            depth = 0
            for name, module in self.named_modules():
                if(type(module) == torch.nn.modules.container.Sequential):
                    for layer in module:
                        # Incrase depth
                        depth += 1
                    
                    # Count reshape "layer"
                    if(name == 'cnn'):
                        depth += 1
            
            return depth - 1
    
    def getDepthMultiBlock(self):
        depth = 0
        for name, module in self.named_modules():
            if(name != '' and name != 'cnn' and name != 'flatten_layer' and name != 'ff'):
                if not (hasattr(module, '__len__')):
                    # Incrase depth
                    depth += 1
                    
            # Count flatten layer
            if(name == 'flatten_layer'):
                depth += 1
        
        return depth - 1
    
    def getMiddleResults(self, x, input_depth, ignore_dropout = True):
        if(self.multi_block_structure):
            return self.getMiddleResultsMultiBlock(x, input_depth, ignore_dropout)
        
        else:
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
    
    def getMiddleResultsMultiBlock(self, x, input_depth, ignore_dropout = True):
        actual_depth = 0
        
        for name, module in self.named_modules():
            if(name != '' and name != 'cnn' and name != 'flatten_layer' and name != 'ff'):
                # print(name, hasattr(module, '__len__'))
                if not (hasattr(module, '__len__')):
                    # Evaluate the value of the input at this level
                    x = module(x)
                    
                    # If I reach the desire level I stop
                    if(actual_depth == input_depth): return x
                    
                    # Incrase depth
                    actual_depth += 1
            
            if(name == 'flatten_layer'):
                x = x.view([x.size(0), -1])
                if(actual_depth == input_depth): return x
                actual_depth += 1
    
    def saveNet(self, path, name_state_dict = "state_dict.pth"):
        # Create path
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save state dict of the network
        torch.save(self.state_dict(), path + name_state_dict)
        
        # Save parameters used to create the network
        # tmp_dict = {'parameters' : self.parameters_creation}
        savemat(path + "parameters.mat", self.parameters_creation)
          
#%%
        
def loadParameters(path):
    parameters = loadmat(path + 'parameters.mat')
    
    # Remove fields added by the savemat function
    parameters.pop('__header__', None)
    parameters.pop('__version__', None)
    parameters.pop('__globals__', None)
    
    # Squeeze extra dimension
    if("activation_list" in parameters.keys()): parameters['activation_list'] = np.squeeze(parameters['activation_list'])
    if("bias_list" in parameters.keys()): parameters['bias_list'] = np.squeeze(parameters['bias_list'])
    if("CNN_normalization_list" in parameters.keys()): parameters['CNN_normalization_list'] = np.squeeze(parameters['CNN_normalization_list'])
    if("dropout_list" in parameters.keys()): parameters['dropout_list'] = np.squeeze(parameters['dropout_list'])
    if("groups_list" in parameters.keys()): parameters['groups_list'] = np.squeeze(parameters['groups_list'])
    
    # Convert single value parameters
    if("h" in parameters.keys()): parameters['h'] = parameters['h'][0,0]
    if("w" in parameters.keys()): parameters['w'] = parameters['w'][0,0]
    if("layers_cnn" in parameters.keys()): parameters['layers_cnn'] = parameters['layers_cnn'][0,0]
    if("layers_ff" in parameters.keys()): parameters['layers_ff'] = parameters['layers_ff'][0,0]
    
    # Convert list of tuple
    if("filters_list" in parameters.keys()): parameters['filters_list'] = convertTupleElementToInt(parameters['filters_list'])
    if("kernel_list" in parameters.keys()): parameters['kernel_list'] = convertTupleElementToInt(parameters['kernel_list'])
    if("padding_list" in parameters.keys()): parameters['padding_list'] = convertTupleElementToInt(parameters['padding_list'])
    
    # Handle pooling list and neurons list
    if("pooling_list" in parameters.keys()): parameters['pooling_list'] = transformPoolingArrayInPoolingList(parameters['pooling_list'])
    if("neurons_list" in parameters.keys()): parameters['neurons_list'] = list(parameters['neurons_list'][0])
    
    return parameters


def transformPoolingArrayInPoolingList(pooling_array):
    pooling_list = []
    pooling_array = np.squeeze(pooling_array)
    
    for element in pooling_array:
        if(element.shape == (1, 1)): pooling_list.append(-1)
        elif (element.shape == (1, 2)):
            element = np.squeeze(element)
            pooling_type = element[0][0, 0]
            pooling_kernel = list(np.squeeze(element[1]))
            pooling_list.append((pooling_type, pooling_kernel))
            
    return pooling_list


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

