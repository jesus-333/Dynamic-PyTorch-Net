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
    if(subject_number == 1): parameters = getSet_1(activation_function)
    elif(subject_number == 2): parameters = getSet_2_8(activation_function)
    elif(subject_number == 3): parameters = getSet_3_4_9(activation_function)
    elif(subject_number == 4): parameters = getSet_3_4_9(activation_function)
    elif(subject_number == 5): parameters = getSet_5(activation_function)
    elif(subject_number == 6): parameters = getSet_6_7(activation_function)
    elif(subject_number == 7): parameters = getSet_6_7(activation_function)
    elif(subject_number == 8): parameters = getSet_2_8(activation_function)
    elif(subject_number == 9): parameters = getSet_3_4_9(activation_function)
    else: parameters = {}
    
    return parameters


def preset(activation_function):
    parameters = {}

    parameters["h"] = 32
    parameters["w"] = 40
    
    parameters["layers_cnn"] = 2 
    parameters["layers_ff"] = 1
    
    i = activation_function
    parameters["activation_list"] = [i, i, 9, 1]
    
    parameters["CNN_normalization_list"] = [True, True]

    parameters["dropout_list"] = [0.5, 0.5, -1, -1]
    
    parameters["neurons_list"] = [4]
    
    return parameters

#%% Group 1 (Subject 1, 3, 4, 6, 7, 9)
# N.b. Pytorch kernel_size, stride, padding, dilation are specifed as height x width


def preset_group_1(activation_function):
    parameters = preset(activation_function)
    parameters["kernel_list"] = [(1, 20), (1, 5)]
    parameters["stride_list"] = [(1, 2), (1, 2)]
    return parameters
    

def getSet_1(activation_function):
    parameters = preset_group_1(activation_function)
    parameters["filters_list"] = [1, 64, 64]
    parameters["filters_list"] = convertArrayInTupleList(parameters["filters_list"])
    return parameters


def getSet_3_4_9(activation_function):
    parameters = preset_group_1(activation_function)
    parameters["filters_list"] = [1, 32, 32]
    parameters["filters_list"] = convertArrayInTupleList(parameters["filters_list"])
    return parameters


def getSet_6_7(activation_function):
    parameters = preset_group_1(activation_function)
    parameters["filters_list"] = [1, 8, 8]
    parameters["filters_list"] = convertArrayInTupleList(parameters["filters_list"])
    return parameters


#%% SET 2 (Subjcet 2 and 8)

def getSet_2_8(activation_function):
    parameters = preset(activation_function)
    parameters["kernel_list"] = [(1, 7), (1, 3)]
    parameters["stride_list"] = [(1, 3), (1, 3)]
    parameters["filters_list"] = [1, 32, 32]
    parameters["filters_list"] = convertArrayInTupleList(parameters["filters_list"])
    return parameters
    

#%% Set 3 (Subject 5)

def getSet_5(activation_function):
    parameters = preset(activation_function)
    parameters["kernel_list"] = [(1, 10), (1, 4)]
    parameters["stride_list"] = [(1, 2), (1, 2)]
    parameters["filters_list"] = [1, 32, 32]
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

