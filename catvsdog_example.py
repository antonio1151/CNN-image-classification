import torch
import numpy as np
#from CNN.mlp import*
from CNN.tools import*
from CNN.model import*
from time import time
import matplotlib.pyplot as plt
import os


parameters={} #initialize parameters

#data setup
parameters['path_cats'] = 'PetImages/Cat'
parameters['path_dogs'] = 'PetImages/Dog'
parameters['size'] =50  #resize the images in r X r
parameters['rebuild'] = True #if this is the fist time running this model, it should be 'True' otherwise False

#setting the convlution neural network
parameters['input_channel'] = 1 #for now the program only works for gray scale but can easily modify
parameters['conv_output_channels'] = [32, 64, 128]
parameters['conv_kernel_size'] = [5, 5, 5]
parameters['stride_size'] = [1, 1, 1]
parameters['padding_size'] = [0, 0, 0]
##Pooling
parameters['pool_kernel_size'] = [2, 2, 2]
parameters['pool_stride_size'] = [2, 2, 2]
parameters['pool_padding_size'] = [0, 0, 0]
#define where the NN will be take place, CPU or GPU
parameters['device']='cuda'

##dimension of the CNN oputput
parameters['output_dimen_cnn'] = size_conv_output(parameters)

#setting full connected NN
parameters['# inputs'] = parameters['output_dimen_cnn']
parameters['# outputs'] = 2
parameters['# layers'] = 3
parameters['# activations'] = parameters['# inputs']
parameters['activation function'] = 'relu' # relu, sigmoid, or LeakyReLU (lrelu)


#define dataset
parameters['validation']=0.1 #porcentage of the datase to be used as a validation set
parameters['training']=1.0-parameters['validation']
parameters['batch_size_training']=100 #number of samples per batch
parameters['batch_size_validation']=100

#training
parameters['learning rate'] = 1.0e-3
parameters['epochs'] = 5 #number of epochs
parameters['max training epochs'] = 1000

#saving model
parameters['savedir'] = 'CNN_catsVsdogs'
parameters['savename'] = 'model'
#define dataset
parameters['dataset']='dataset.npy'


#loading the model
cnn = model(parameters)

if parameters['rebuild']:
#loading training and validation datasets
    training, validation = load_data_catsVsdogs(parameters)
#train the model
    cnn.converge(training, validation)
    cnn.visual_loss()
    cnn.save_model()
    os.replace(parameters['dataset'],'{}/{}'.format(parameters['savedir'],parameters['dataset']))
    os.replace('PetImages','{}/PetImages'.format(parameters['savedir']))
else:

    cnn.load_model()

#use the model to determine if an image is/has a dog or cat

path = 'kitty.jpg'  #here you put the path and name of the image you want to classify
cnn.pred_catVsdog(path)
