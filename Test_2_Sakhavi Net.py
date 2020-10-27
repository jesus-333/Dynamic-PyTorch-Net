import numpy as np

import torch

from jesus_network_V2 import DynamicCNN, convertArrayInTupleList

#%% SET 1

print_var = True

parameters = {}

parameters["h"] = 32
parameters["w"] = 40

parameters["layers_cnn"] = 2 
parameters["layers_ff"] = 1

parameters["activation_list"] = [1, 1, 1, 1]

parameters["kernel_list"] = [(1, 7), (1, 3)]

parameters["filters_list"] = [1, 32, 32]
parameters["filters_list"] = convertArrayInTupleList(parameters["filters_list"])

parameters["stride_list"] = [(1, 3), (1, 3)]

parameters["neurons_list"] = [512, 4]
parameters["neurons_list"] = convertArrayInTupleList(parameters["neurons_list"])


model = DynamicCNN(parameters, print_var)

x_test = torch.ones((1, 1, parameters["h"], parameters["w"]))
y_test = model(x_test)

print(model, "\n\n\n")

# for parameter in model.parameters():
#     print(parameter.numel())