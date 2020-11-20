# -*- coding: utf-8 -*-
"""


@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

#%% Add support function folder to path
import torch
from DynamicNet import DynamicCNN, convertArrayInTupleList

#%%

def getSubjectParameters(subject_number, activation_function = 0):
    # print(subject_number)
    if  (subject_number == 1): parameters = getSet_1_3(activation_function)
    elif(subject_number == 2): parameters = getSet_2(activation_function)
    elif(subject_number == 3): parameters = getSet_1_3(activation_function)
    elif(subject_number == 4): parameters = getSet_4(activation_function)
    elif(subject_number == 5): parameters = getSet_5(activation_function)
    elif(subject_number == 6): parameters = getSet_6(activation_function)
    elif(subject_number == 7): parameters = getSet_7(activation_function)
    elif(subject_number == 8): parameters = getSet_8(activation_function)
    elif(subject_number == 9): parameters = getSet_9(activation_function)
    else: parameters = {}
    
    # n_fitlers = parameters["filters_list"][0][1]
    # parameters["groups_list"] = [1, n_fitlers, n_fitlers]
    
    return parameters


def preset(activation_function):
    parameters = {}

    parameters["h"] = 32
    parameters["w"] = 40
    
    parameters["layers_cnn"] = 3 
    parameters["layers_ff"] = 1
    
    i = activation_function
    parameters["activation_list"] = [i, i, i, 9, 1]
    # parameters["activation_list"] = [2, 2, i, 9, 1]
    
    parameters["CNN_normalization_list"] = [True, True, True]

    parameters["dropout_list"] = [0.5, 0.5, 0.5, -1, -1]
    
    parameters["neurons_list"] = [4]
    
    return parameters

#%% Group 1 (Subject 1, 3,)
# N.b. Pytorch kernel_size, stride, padding, dilation are specifed as height x width

def getSet_1_3(activation_function):
    parameters = preset(activation_function)
    parameters["kernel_list"] = [(1, 10), (1, 4), (32, 1)]
    parameters["stride_list"] = [(1, 2), (1, 2), (32, 1)]
    parameters["filters_list"] = [1, 256, 256, 256]
    parameters["filters_list"] = convertArrayInTupleList(parameters["filters_list"])
    return parameters
    


#%% Group 2 (Subjcet 2, 7, 8)

def preset_group_2(activation_function):
    parameters = preset(activation_function)
    parameters["kernel_list"] = [(1, 4), (1, 3), (32, 1)]
    parameters["stride_list"] = [(1, 3), (1, 2), (1, 1)]
    return parameters
    

def getSet_2(activation_function):
    parameters = preset_group_2(activation_function)
    parameters["filters_list"] = [1, 32, 32, 32]
    parameters["filters_list"] = convertArrayInTupleList(parameters["filters_list"])
    return parameters


def getSet_7(activation_function):
    parameters = preset_group_2(activation_function)
    parameters["filters_list"] = [1, 256, 256, 256]
    parameters["filters_list"] = convertArrayInTupleList(parameters["filters_list"])
    return parameters


def getSet_8(activation_function):
    parameters = preset_group_2(activation_function)
    parameters["filters_list"] = [1, 8, 8, 8]
    parameters["filters_list"] = convertArrayInTupleList(parameters["filters_list"])
    return parameters

#%% Group 3 (Subjcet 4, 5, 6)

def preset_group_3(activation_function):
    parameters = preset(activation_function)
    parameters["kernel_list"] = [(1, 20), (1, 5), (32, 1)]
    parameters["stride_list"] = [(1, 2), (1, 2), (1, 1)]
    return parameters
    

def getSet_4(activation_function):
    parameters = preset_group_3(activation_function)
    parameters["filters_list"] = [1, 256, 256, 256]
    parameters["filters_list"] = convertArrayInTupleList(parameters["filters_list"])
    return parameters


def getSet_5(activation_function):
    parameters = preset_group_3(activation_function)
    parameters["filters_list"] = [1, 128, 128, 128]
    parameters["filters_list"] = convertArrayInTupleList(parameters["filters_list"])
    return parameters


def getSet_6(activation_function):
    parameters = preset_group_3(activation_function)
    parameters["filters_list"] = [1, 32, 32, 32]
    parameters["filters_list"] = convertArrayInTupleList(parameters["filters_list"])
    return parameters
    

#%% Group 4 (Subject 9)

def getSet_9(activation_function):
    parameters = preset(activation_function)
    parameters["kernel_list"] = [(1, 10), (1, 3), (32, 1)]
    parameters["stride_list"] = [(1, 3), (1, 2), (1, 1)]
    parameters["filters_list"] = [1, 256, 256, 256]
    parameters["filters_list"] = convertArrayInTupleList(parameters["filters_list"])
    return parameters


#%%
print_var = False
tracking_input_dimension = False

idx = 2
parameters = getSubjectParameters(idx)

model = DynamicCNN(parameters, print_var, tracking_input_dimension = tracking_input_dimension)

x_test = torch.ones((1, 1, parameters["h"], parameters["w"]))
y_test = model(x_test)

print(model, "\n\n\n")

