# Dynamic PyTorch CNN

I this repository I upload a class that allow to create in automatic a Convolutional-Neural-Network (CNN) or a Feed-Forward-Neural-Network in PyTorch.

## Brief Introduction and Motivation
Why create somthing similar? Well basically becasue it didn't exist (or at least I can't find something similar). With this class you can simply create a neural network specifying only a dictionary of parameter. 

In this way the design of the network can be far more easy, especially for begginer that might just want create their first network. But also if you are an expert and you're bored to write every time the entire class, specify the forward method, the constructor etc etc. Or maybe you have to test various network that differ only for a few parameter... in this case you only change some component of the dictionary and the work is done.

Keep in mind that this is a relative simple project. You can create (at least for now) only standard convolutional and feed-forward network. So no convolutional autoencoder (autoencoder with only feed-forward network can be created), GANs, Boltzman Machine or network with multiple branch. 

![What can you do?](master/docs/scheme_what you can do.jpg)
