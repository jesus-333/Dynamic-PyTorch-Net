import numpy as np

import torch

from DynamicNet import DynamicCNN, convertArrayInTupleList

#%%

print_var = False
tracking_input_dimension = True

C = 32
T = 1024

F_1 = 8
D = 2
F_2 = 16

kernel_1 = (1, 64)
kernel_2 = (C, 1)
kernel_3 = (1, 16)
kernel_4 = (1, 1)

parameters = {}

parameters["h"] = C
parameters["w"] = T

parameters["layers_cnn"] = 4 
parameters["layers_ff"] = 0

parameters["activation_list"] = [-1, 0, -1, 8, 9]

parameters["kernel_list"] = [kernel_1, kernel_2, kernel_3, kernel_4]

parameters["filters_list"] = [1, F_1, F_1 * D, F_1 * D, F_2]
parameters["filters_list"] = convertArrayInTupleList(parameters["filters_list"])


parameters["padding_list"] = [(0, int(kernel_1[1]/2)), [0,0], (0, int(kernel_3[1]/2)), [0,0]]

parameters["CNN_normalization_list"] = [True, True, False, True]

parameters["dropout_list"] = [-1, 0.25, -1, 0.25, -1]

parameters["pooling_list"] = [-1, [1, (1,4)], -1, [1, (1,8)]]

parameters["groups_list"] = [1, F_1, F_1 * D, 1]


model = DynamicCNN(parameters, print_var, tracking_input_dimension = tracking_input_dimension)

x_test = torch.ones((1, 1, parameters["h"], parameters["w"]))
y_test = model(x_test)

# print(model)

#%%

# for name, param in model.named_parameters():
#     print(name, param.size())

for parameter in model.parameters():
    print(parameter.shape)