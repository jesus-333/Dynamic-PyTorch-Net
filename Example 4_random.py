import numpy as np

import torch

from DynamicNet import DynamicCNN, convertArrayInTupleList

import random

#%% SET 1

print_var = True
tracking_input_dimension = True

parameters = {}

parameters["h"] = random.randint(80, 100)
parameters["w"] = random.randint(80, 100)

parameters["layers_cnn"] = random.randint(1, 4)
parameters["layers_ff"] = random.randint(1, 4)
parameters["layers_ff"] = 4

# parameters["h"] = 100
# parameters["w"] = 100

parameters["layers_cnn"] = 0
# parameters["layers_ff"] = 0

if(parameters["layers_cnn"] > 0): 
    parameters["activation_list"] = np.random.randint(0, 10, parameters["layers_cnn"] + parameters["layers_ff"] + 1)
else: 
    parameters["activation_list"] = np.random.randint(0, 10, parameters["layers_ff"])

parameters["kernel_list"] = []
parameters["filters_list"] = [1]
parameters["stride_list"] = []
for i in range(parameters["layers_cnn"]):
    parameters["kernel_list"].append((random.randint(1, 5), random.randint(1, 5)))
    parameters["filters_list"].append(random.randint(2, 64))
    parameters["stride_list"].append((random.randint(1, 3), random.randint(1, 3)))
                                      
parameters["filters_list"] = convertArrayInTupleList(parameters["filters_list"])

parameters["neurons_list"] = []
for i in range(parameters["layers_ff"]):
    parameters["neurons_list"].append(random.randint(2, 128))
    print(parameters["neurons_list"])


model = DynamicCNN(parameters, print_var, tracking_input_dimension = tracking_input_dimension)

if(parameters["layers_cnn"] > 0):
    x_test = torch.ones((1, 1, parameters["h"], parameters["w"]))
else:     
    x_test = torch.ones((1,parameters["neurons_list"][0]))
                          
y_test = model(x_test)

print(model, "\n\n\n")

# for parameter in model.parameters():
#     print(parameter.numel())