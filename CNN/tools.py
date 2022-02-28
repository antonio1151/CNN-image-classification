import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import numpy as np
import cv2
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms, datasets
from zipfile import ZipFile
import wget

class buildDataset():
#read and organize the sets (training and validation)
    def __init__(self, params):
        self.params = params
        self.dataset = np.load(self.params['dataset'],allow_pickle=True)
        self.samples = self.dataset[:,0]
        self.targets = self.dataset[:,1]


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]

    def split_data(self):
        self.size_valid = int(round(self.params['validation']*len(self.dataset[:,0])))
        self.size_train = int(round(self.params['training']*len(self.dataset[:,0])))
        return random_split(self,[self.size_train,self.size_valid])

#general loading data function
def load_data(parameters):
    params = parameters
    dataset = buildDataset(params)
    train, valid = dataset.split_data()
    train_dl=DataLoader(train,batch_size=params['batch_size_training'], shuffle=True, pin_memory=True)
    valid_dl=DataLoader(valid,batch_size=params['batch_size_validation'], shuffle=True, pin_memory=True)
    return  train_dl, valid_dl

#loading data function for cat vs dog model
def load_data_catsVsdogs(parameters):
    params = parameters
    if params['rebuild']:
        dat=data_of_inmages(parameters)
        dat.creates_dataset()
    dataset = buildDataset(params)
    train, valid = dataset.split_data()
    train_dl=DataLoader(train,batch_size=params['batch_size_training'], shuffle=True, pin_memory=True)
    valid_dl=DataLoader(valid,batch_size=params['batch_size_validation'], shuffle=True, pin_memory=True)
    return  train_dl, valid_dl


#plot losses
def plot(loss_tr,loss_val):
    plt.plot(loss_tr,label='loss training')
    plt.plot(loss_val,label = 'loss validation')
    plt.legend()
    plt.grid()
    plt.show()


##load image data for cats and dog
class data_of_inmages():
    def __init__(self,params):
        self.path_cats = params['path_cats']
        self.path_dogs = params['path_dogs']
        self.size = params['size']
        self.namedata=params['dataset']
        #extract the dataset
        print("downloading the dataset")
        wget.download("https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip")
        with ZipFile('kagglecatsanddogs_3367a.zip', 'r') as zipObj:
           # Extract all the contents of zip file in current directory
           zipObj.extractall()
        os.remove("kagglecatsanddogs_3367a.zip")
    #creates hot vector for the prediction
        self.labels = {self.path_cats:0, self.path_dogs:1}
    #counter and list for the data
        self.data = []
        self.catscount = 0
        self.dogscount = 0

    def creates_dataset(self):
        for namedir in self.labels:
            for ima in tqdm(os.listdir(namedir)):
                if ima.endswith('.jpg'):
                    try:
                        path = os.path.join(namedir,ima)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) #transforms the image in gray scale
                        #resize the image
                        img = cv2.resize(img, (self.size, self.size))
                        #creates the array of image matrix and hot vector
                        self.data.append([np.array(img)/255.0,np.eye(2)[self.labels[namedir]]]) #normalize the image

                        if namedir == self.path_cats:
                            self.catscount += 1
                        elif namedir == self.path_dogs:
                            self.dogscount += 1

                    except Exception as e:
                        pass

        np.random.shuffle(self.data)
        np.save(self.namedata, self.data)
        print('# of sample of cats=',self.catscount)
        print('# of sample of dogs=',self.dogscount)

###getting convolution nn output size
def size_conv_output(params):
    sizee=params['size']
#    kernel_conv=params['conv_kernel_size']
#    padding_conv=params['padding_size']
    for i in range(len(params['conv_output_channels'])):
        w_conv=int((sizee-params['conv_kernel_size'][i]+2*params['padding_size'][i])/(params['stride_size'][i])+1)
        w_pool=int((w_conv-params['pool_kernel_size'][i])/(params['pool_stride_size'][i])+1)
        sizee=w_pool
    return sizee*sizee*params['conv_output_channels'][-1]

def new_image(path,size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size, size))
    return np.array(img)/255.0
