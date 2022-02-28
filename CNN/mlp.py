import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class activation(nn.Module):
        def __init__(self, activation_func):
            super().__init__()
            if activation_func == 'relu':
                self.activation = nn.ReLU()
            elif activation_func =='sigmoid':
                self.activation = nn.Sigmoid()
            elif activation_func == 'lrelu':
                self.activation = nn.LeakyReLU()


        def forward(self, input):
            return self.activation(input)# model definition

class MLP(nn.Module):
    def __init__(self,params):
        super(MLP,self).__init__()
        self.params = params
        self.device = torch.device(self.params['device'])
########convlution setting
        self.in_out_conv = self.params['conv_output_channels']
        self.kernel_conv = self.params['conv_kernel_size']
        self.stride_conv = self.params['stride_size']
        self.padding_conv = self.params['padding_size']
        self.kernel_pool = self.params['pool_kernel_size']
        self.stride_pool = self.params['pool_stride_size']
        self.padding_pool = self.params['pool_padding_size']
        self.initial_conv = nn.Conv2d(self.params['input_channel'],self.in_out_conv[0],
                         self.kernel_conv[0], stride=self.stride_conv[0], padding=self.padding_conv[0],
                         device=self.device,dtype=torch.float64)
        self.convnn = []
        self.maxpoolnn = []
        self.activ_conv = []

        for i in range(len(self.in_out_conv)-1):
            self.convnn.append(nn.Conv2d(self.in_out_conv[i],self.in_out_conv[i+1],
            self.kernel_conv[i], stride=self.stride_conv[i], padding=self.padding_conv[i],
            device=self.device,dtype=torch.float64))


        for j in range(len(self.kernel_pool)):
            self.maxpoolnn.append(nn.MaxPool2d(self.kernel_pool[j], stride= self.stride_pool[0],
                                             padding=self.padding_pool[j]))
            self.activ_conv.append(activation('relu'))

        self.convnn = nn.ModuleList(self.convnn)
        self.maxpoolnn =  nn.ModuleList(self.maxpoolnn)
        self.activ_conv=  nn.ModuleList(self.activ_conv)

#######setting fully connected NN
        self.n_inputs = self.params['# inputs']
        self.n_outputs = self.params['# outputs']
        self.n_layer = self.params['# layers']
        self.n_activation = self.params['# activations']
        self.device = torch.device(self.params['device'])
        #building the input and output layers
        self.initial_layer = nn.Linear(self.n_inputs,self.n_activation,device=self.device,dtype=torch.float64)
        self.activation1 = activation(self.params['activation function'])
        self.output_layer = nn.Linear(self.n_activation,self.n_outputs,device=self.device,dtype=torch.float64)
        #build the Neural network
        self.lin_layers=[]
        self.activations=[]

        for i in range(self.n_layer):
            self.lin_layers.append( nn.Linear(self.n_activation,self.n_activation,device=self.device,dtype=torch.float64))
            self.activations.append(activation(params['activation function']))

        # initialize module lists
        self.lin_layers = nn.ModuleList(self.lin_layers)
        self.activations = nn.ModuleList(self.activations)


    #convolution neural network
    def conv(self,x):
        x = self.initial_conv(x)
        x = self.activ_conv[0](x)
        x = self.maxpoolnn[0](x)
        for i in range(len(self.kernel_pool)-1):
            x = self.convnn[i](x)
            x = self.activ_conv[i+1](x)
            x = self.maxpoolnn[i+1](x)
        return x

    #forward propagate input
    def forward(self,x):
        x = self.conv(x)
        x = x.view(-1,self.params['# inputs'])
        x = self.activation1(self.initial_layer(x))
        for i in range(self.n_layer):
            x = self.lin_layers[i](x)
            x = self.activations[i](x)

        x=self.output_layer(x)
        return F.softmax(x,dim=1)
