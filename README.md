# Dynamic PyTorch Neural Networks or DynamicNet

I this repository I upload a class that allow to create in automatic a Convolutional-Neural-Network (CNN) or a Feed-Forward-Neural-Network in PyTorch. I upload a short video demonstration on YouTube at this [link](https://youtu.be/hS7DnarmL-M).

## Brief Introduction and Motivation
Why create somthing similar? Well basically becasue it didn't exist (or at least I can't find something similar). With this class you can simply create a neural network specifying only a dictionary of parameter. 

In this way the design of the network can be far more easy, especially for begginer that might just want create their first network. But also if you are an expert and you're bored to write every time the entire class, specify the forward method, the constructor etc etc. Or maybe you have to test various network that differ only for a few parameter... in this case you only change some component of the dictionary and the work is done.

Keep in mind that this is a relative simple project. You can create (at least for now) only standard convolutional and feed-forward network. So no convolutional autoencoder (autoencoder with only feed-forward network can be created), GANs, Boltzman Machine or network with multiple branch. The image below show the type of network you can create with my script. **If you want you can create single part of this more complicated network with my tool and then stick all together inside a new class**.

![Which network can you create?](https://github.com/jesus-333/Dynamic-PyTorch-CNN/blob/main/docs/scheme_what_you_can_do.jpg)


## Tutorial
In this section I will explain how to use my class to create your network. All the class is based on receiving a dictionary (that I will call *parameters*) as input. I will divide the section in three parts: 1) *parameters* for the convolutional part of the network 2) *parameters* for the feed forward part of the network 3) *parameters* in common between the two parts. Each section will be structured as list of keys for the dicionary. The first word in italic will be the name of the key and then a description will follow.


### Convolutional Parameters
* *layers_cnn*: pretty straightforward. This is the number of convolutional layer. Must be an int. If set to 0 the network will not have a convolutional section. It cannot be 0 if also *layers_ff* is 0.
* *kernel_list*: list of tuple. Each tuple represent the dimension of the kernel for the current layer. Each kernel must be specified as (heigh, width). It must have a length equals to *layers_cnn*.
* *filters_list*: list of tuple. Each tuple represent the number of input channel and output channel for the layer. For example if you use BGR image as input and you want 9 channel at the end of the first convolution you must write the tuple as (3,9). In case you're lazy (like me) I provided a support function called *convertArrayInTupleList()* that you can use to convert a list of int in the list of tuple necessary for the network. In this case you only need to specify the output filter for each layer (plus the number of input channel). So for example, always with a BGR image as input, you want to perform 3 convolution with, respectively, 9, 27 and 54 as outuput filters. So you can write your *filters_list* as [3, 9, 27, 54]  and the function will convert your list in [(3, 9), (9, 27), (27, 54)]. It must have a length equals to *layers_cnn*.
* (OPTIONAL) *stride_list*: list of tuple. If you don't know what it is stride don't bother and left it empty. Each tuple represent the stride of the kernel for the current layer. Each tuple must be specified as (stride in height, stride in width). It must have a length equals to *layers_cnn*.
* (OPTIONAL) *padding_list*: list of tuple. If you don't know what it is padding don't bother and left it empty. Each tuple represent the padding for the current layer. Each tuple must be specified as (padding in height, padding in width). It must have a length equals to *layers_cnn*.
* (OPTIONAL) *pooling_list*: list of int and tuple. If you don't know what it is polling don't bother and left it empty. Each element of the list represent the pooling for the current layer. If you don't want any pooling in the current layer set the value to -1. Otherwise the input must be in the following form: (*n* , (x, y)) where n represent the type of pooling and (x,y) the kernel of the pooling. *n* = 0 is for the MaxPool2D while *n* = 1 is for the AvgPool2D. It must have a length equals to *layers_cnn*.
* (OPTIONAL) *groups_list*: list of int. If you don't know what it is group don't bother and left it empty. Each int represent the group for the current layer. It must have a length equals to *layers_cnn*.
* (OPTIONAL) *CNN_normalization_list*: list of bool. Each bool indicate if executing or not a normalization for the output of the convolution of the current layer. It must have a length equals to *layers_cnn*.
* (OPTIONAL) *add_flatten_layer* bool. In case you create a network with only a cnn section this parameter choose if add or not the flatten layer at the end. By defualt is set to True.

N.B. You can avoid to create all the field with (OPTIONAL) if you don't bother. Also if *layers_cnn* is set to 0 you don't have to specify any of this parameters.

### Feed-Forward Parameters 
* *layers_ff*: pretty straightforward. This is the number of feed-forward layer. Must be an int. If set to 0 the network will not have a feed-forward section. It cannot be 0 if also *layers_cnn* is 0.
* *neurons_list*: list of int. Each number represents the number of neurons for the current layer of the network. The input neurons of the fully connect layer between the cnn and the feedforward part are evaluated in automatic by the class.

**Particular case of feed-forward**: 

### Common Parameters
* *activation_list*: lis of int. Each number represent the activation function for the current layer. Generally it must have a length of *layers_cnn* + *layers_ff* + 1. The +1 is derive form the fact that when you flatten the output of the convolutional section and attach that flatten outuput to the feed-forward section you create a new layer in the feed-forward section. In case you need only the cnn section ist must have the length *layers_cnn* (*add_flatten_layer* = False) or *layers_cnn + 1* (*add_flatten_layer* = True). In case you need only the feed-forward section ist must have the length *layers_ff*. The possible values are (you can add more activation modifying the function *getActivationList()* in the *support_DynamicNet.py* file):
  * -1: No activation.
  * 0: ReLU (Rectified Linear Unity).
  * 1: Leaky ReLU.
  * 2: SELU (Scaled Exponential Linear Unit).
  * 3: ELU (Exponential Linear Unit).
  * 4: GELU (Gaussian Error Linear Units)
  * 5: Sigmoid
  * 6: Tanh
  * 7: Hardtanh
  * 8: Hardshrink
  * 9: LogSoftmax
  * 10: Softmax
* *dropout_list*: list of float. Specify the dropout probability for each layer. If the entry for the corresponding layer is -1 no dropout will be used. It must have a length of *layers_cnn* + *layers_ff* + 1. In case you need only the cnn section ist must have the length *layers_cnn* (*add_flatten_layer* = False) or *layers_cnn + 1* (*add_flatten_layer* = True). In case you need only the feed-forward section ist must have the length *layers_ff*.
* *bias_list*: list of bool. Specify if use or not the bias for the current layer (for both cnn section and feed-forward section). It must have a length of *layers_cnn* + *layers_ff* + 1. In case you need only the cnn section ist must have the length *layers_cnn* (*add_flatten_layer* = False) or *layers_cnn + 1* (*add_flatten_layer* = True). In case you need only the feed-forward section ist must have the length *layers_ff*.
  
N.B. For more info about the various activation function or search for new activaion read this [link](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity).

## Network and Features visualization
The class is also provided with a method called *printNetwork()* that print all the structure of the network showing also the "depth" of each layer (pay attention that every transformation of the input will increase the depth of one... for example if we have a convolution followed by a normalizationa and an activation the convolution will be at depth 0, the normalization at depth 1 and the activation at depth 2).

Also, if you want ot obtain the input after be passed through only a part of the network, the class provide a the *getMiddleResults()* method. With this method you send in input a data *x*, specify the depth (see previous paragraph) and you will obtain how the input will be at that level of depth.

## Examples
I provided 4 different files where I create some network with my *DynamicNet*. In the firs I recreate the [EEGNet](https://arxiv.org/abs/1611.08024) a famouse network used to analyze EEG signal. In the second and the third I create the networks presented in the work of [Sakhavi et al.](https://ieeexplore.ieee.org/document/8310961). In the last file I simple set up everything at random and a random network will be created.

N.B. The file must be in the same folder of *DynamicNet.py* and *support_DynamicNet.py*  

# Citation
If you use this project please cite me and share this repository :pray: :heart:
