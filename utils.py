#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:13:16 2020

@author: michael
"""


import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import copy
import pyprind

import numpy as np
import os
import pickle

import sys
sys.path.append('/home/michael/Documents/Scxript/torchsample-master/torchsample/')
#from skimage.transform import rescale, resize, downscale_local_mean, rotate
#import skimage.segmentation as seg
#from skimage.transform import swirl
#import matplotlib.pyplot as plt 

cwd = os.getcwd()

def conv1d_block_calculator(input_w, input_h = None, kernel = 5, stride = 0, padding = 0, pooling = 0):
    if not input_h:
        input_h = input_w
    dim_w = ((input_w + 2*padding - (kernel-1)-1)/stride) + 1
    dim_h = ((input_h + 2*padding - (kernel-1)-1)/stride) + 1
    if pooling:
        dim_w /=pooling
        dim_h /=pooling
    return np.floor(dim_w), np.floor(dim_h) 

def create_train_test(dr = cwd):
    with open(cwd+'/mnist_padded_act_full1.pkl', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data1 = pickle.load(f)
    
    with open(cwd+'/mnist_padded_act_full2.pkl', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data2 = pickle.load(f)
    
    
    data = data1[0] + data2[0]
    labels = data1[1] + data2[1]
    dataset = [data,labels]
    dataset = one_hot_dataset(dataset)
    
    data1 = []
    data2 = []
    
    k = 0.1 #size of test set
    train, test = sampleFromClass(dataset,k)
    return train, test

def sampleFromClass(ds, k):
    '''
    ds == Dataset
    k == Number of examples from each class
    '''
    class_counts = {}
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    unique, count = np.unique(ds.targets.numpy(), return_counts = True)
    count_dict = {}
    for i in range(len(unique)):
        count_dict[i] = count[i]*(1-k)
        
    for data, label in ds:
        
        c = label.item()
        class_counts[c] = class_counts.get(c, 0) + 1
        if class_counts[c] < count_dict[i]:
            train_data.append(data)
            train_label.append(torch.unsqueeze(label, 0))
        else:
            test_data.append(data)
            test_label.append(torch.unsqueeze(label, 0))
    print(class_counts)
    train_data = torch.stack(train_data)
    #for ll in train_label:
    #    print(ll)
    train_label = torch.cat(train_label)
    test_data = torch.stack(test_data)
    test_label = torch.cat(test_label)
    print(train_data.size(), train_label.size(),test_data.size(),test_label.size())
    return (torch.utils.data.TensorDataset(train_data, train_label), 
        torch.utils.data.TensorDataset(test_data, test_label))
    
class one_hot_dataset(Dataset):
    def __init__(self, traject_data, nb_digits = 9):
        
        self.targets = torch.tensor(traject_data[1])
        self.orig_data = traject_data[0]
        self.data = []
        bar = pyprind.ProgBar(len(self.orig_data),monitor = True)
        for i in range(len(self.orig_data)):
            self.data.append(one_hot(self.orig_data[i]).transpose(0,1))
            bar.update()
        self.data = torch.stack(self.data)
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        '''
        args idx (int) :  index
        
        returns: tuple(trajectory, label)
        '''
        trajectory = self.data[idx]
        label = self.targets[idx]
        
        return trajectory, label
    
    def dataset(self):
        return self.data
    def labels(self):
        return self.targets
    

def one_hot(input_data, nb_digits = 9):

    if type(input_data) != torch.Tensor:
        input_data = torch.tensor(input_data)
    y = input_data.long().unsqueeze(1) % nb_digits
    #One hot encoding buffer that you create out of the loop and just keep reusing
    
    y_onehot = torch.FloatTensor(len(input_data), nb_digits)
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    
    return y_onehot 


def traject_one_hot():
    with open(cwd+'/mnist_padded_act_full1.pkl', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data1 = pickle.load(f)
    
    with open(cwd+'/mnist_padded_act_full2.pkl', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data2 = pickle.load(f)
    
    
    data = data1[0] + data2[0]
    labels = data1[1] + data2[1][:len(data2[0])]
    dataset = [data,labels]
    dataset = one_hot_dataset(dataset)
    
    return dataset
#Build Dataset of trajectories and MNIST
#Checked and the trajectories order is the same as the data set. 
class MNIST_TRAJECT_MINE(Dataset):
    def __init__(self, transform = None):
        
        train = True
        self.transform = transform
        
        #Creating trajectory, marginal_mnist and joint_mnist disterbution datasets for MINE
        #Example for an instance: 
        #trajectory:label=3, joint:label=3(different image), marginal:label=5(or other not 3)
        
        trajectory_dataset = traject_one_hot()
        
        for i in range(10):

            #temp_traject = copy.deepcopy(trajectory_dataset.targets)
            mnist_dataset = torchvision.datasets.MNIST(os.getcwd(),
                    train = train, download = True)
                    
            #creating the main data set arranged by the label, shuffle SHOULD 
            #be done while calling dataloader.
            idx = trajectory_dataset.targets==i
            temp_targets = trajectory_dataset.targets[idx]
            temp_data = trajectory_dataset.data[idx]
            if i == 0:
                self.traject_data = temp_data
                self.traject_targets = temp_targets
            else:
                self.traject_data = torch.utils.data.ConcatDataset([self.traject_data, temp_data])
                self.traject_targets = torch.utils.data.ConcatDataset([self.traject_targets, temp_targets])
              
            #Creating marginal dataset, i.e. a random image drown from the pool
            idx = np.random.randint(0,len(mnist_dataset),len(temp_targets))
            marginal_temp = torchvision.datasets.MNIST(os.getcwd(),train = train, download = True)
            marginal_temp.targets = mnist_dataset.targets[idx]
            marginal_temp.data = mnist_dataset.data[idx]
            if i == 0:
                self.marginal_data = marginal_temp.data
                self.marginal_targets = marginal_temp.targets
            else:
                self.marginal_data = torch.utils.data.ConcatDataset([self.marginal_data, marginal_temp.data])
                self.marginal_targets = torch.utils.data.ConcatDataset([self.marginal_targets, marginal_temp.targets])
            
            #Creating a joint disterbution dataset, i.e. same label different
            #image 
            idx = mnist_dataset.targets==i
            mnist_dataset.targets = mnist_dataset.targets[idx]
            mnist_dataset.data = mnist_dataset.data[idx]
            idx = torch.randperm(len(mnist_dataset))
            mnist_dataset.targets = mnist_dataset.targets[idx]
            mnist_dataset.data = mnist_dataset.data[idx]
            if i == 0:
                self.sec_data = mnist_dataset.data
                self.sec_targets = mnist_dataset.targets
            else:
                self.sec_data = torch.utils.data.ConcatDataset([self.sec_data, mnist_dataset.data])
                self.sec_targets = torch.utils.data.ConcatDataset([self.sec_targets, mnist_dataset.targets])
            
                
           
                   
    
    def __len__(self):
        return len(self.traject_data)
    
    def __getitem__(self, idx):
        '''
        args idx (int) :  index
        
        returns: tuple(main_data, main_target, sec_joint, sec_joint_target, sec_matginal, sec_marginal_target)
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        main_data = self.main_data[idx]
        main_targets = self.main_targets[idx]
        
        sec_joint_data = self.sec_data[idx]
        sec_joint_targets = self.sec_targets[idx]
        
        sec_marginal_data = self.marginal_data[idx]
        sec_marginal_targets = self.marginal_targets[idx]
            
        
        if self.transform is not None:
            main_data = self.transform(main_data)
            sec_joint_data = self.transform(sec_joint_data)
            sec_marginal_data = self.transform(sec_marginal_data)
            
        

        return main_data, main_targets, sec_joint_data, sec_joint_targets, sec_marginal_data, sec_marginal_targets
    
    def targets(self):
        return self.main_targets, self.sec_targets, self.marginal_targets
    
    def data(self):
        return self.main_data, self.sec_data, self.marginal_data
    
    
class MNIST_for_MINE(Dataset):
    def __init__(self, train = False, transform = None):
        
        self.transform = transform
        
        #Creating main, marginal and joint disterbution datasets for MINE
        #Example for an instance: 
        #main:label=3, joint:label=3(different image), marginal:label=5(or other not 3)
        for i in range(10):
            main_dataset = torchvision.datasets.MNIST(os.getcwd(),
                    train = train,download = True)
            sec_dataset = torchvision.datasets.MNIST(os.getcwd(),
                    train = train, download = True)
                    
            #creating the main data set arranged by the label, shuffle SHOULD 
            #be done while calling dataloader.
            idx = main_dataset.targets==i
            main_dataset.targets = main_dataset.targets[idx]
            main_dataset.data = main_dataset.data[idx]
            if i == 0:
                self.main_data = main_dataset.data
                self.main_targets = main_dataset.targets
            else:
                self.main_data = torch.utils.data.ConcatDataset([self.main_data, main_dataset.data])
                self.main_targets = torch.utils.data.ConcatDataset([self.main_targets, main_dataset.targets])
              
            #Creating marginal dataset, i.e. a random image drown from the pool
            idx = np.random.randint(0,len(sec_dataset),len(main_dataset))
            marginal_temp = torchvision.datasets.MNIST(os.getcwd(),train = train, download = True)
            marginal_temp.targets = sec_dataset.targets[idx]
            marginal_temp.data = sec_dataset.data[idx]
            if i == 0:
                self.marginal_data = marginal_temp.data
                self.marginal_targets = marginal_temp.targets
            else:
                self.marginal_data = torch.utils.data.ConcatDataset([self.marginal_data, marginal_temp.data])
                self.marginal_targets = torch.utils.data.ConcatDataset([self.marginal_targets, marginal_temp.targets])
            
            #Creating a joint disterbution dataset, i.e. same label different
            #image 
            idx = sec_dataset.targets==i
            sec_dataset.targets = sec_dataset.targets[idx]
            sec_dataset.data = sec_dataset.data[idx]
            idx = torch.randperm(len(sec_dataset))
            sec_dataset.targets = sec_dataset.targets[idx]
            sec_dataset.data = sec_dataset.data[idx]
            if i == 0:
                self.sec_data = sec_dataset.data
                self.sec_targets = sec_dataset.targets
            else:
                self.sec_data = torch.utils.data.ConcatDataset([self.sec_data, sec_dataset.data])
                self.sec_targets = torch.utils.data.ConcatDataset([self.sec_targets, sec_dataset.targets])
            
                
           
                   
    
    def __len__(self):
        return len(self.main_data)
    
    def __getitem__(self, idx):
        '''
        args idx (int) :  index
        
        returns: tuple(main_data, main_target, sec_joint, sec_joint_target, sec_matginal, sec_marginal_target)
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        main_data = self.main_data[idx]
        main_targets = self.main_targets[idx]
        
        sec_joint_data = self.sec_data[idx]
        sec_joint_targets = self.sec_targets[idx]
        
        sec_marginal_data = self.marginal_data[idx]
        sec_marginal_targets = self.marginal_targets[idx]
            
        
        if self.transform is not None:
            main_data = self.transform(main_data)
            sec_joint_data = self.transform(sec_joint_data)
            sec_marginal_data = self.transform(sec_marginal_data)
            
        

        return main_data, main_targets, sec_joint_data, sec_joint_targets, sec_marginal_data, sec_marginal_targets
    
    def targets(self):
        return self.main_targets, self.sec_targets, self.marginal_targets
    
    def data(self):
        return self.main_data, self.sec_data, self.marginal_data
'''   
def noisy(noise_typ,image, rotation = 45, swirl_rate = 5):
    
    #Parameters
    #----------
    #image : ndarray
    #Input image data. Will be converted to float.
    #mode : str
    #One of the following strings, selecting the type of noise to add:
    #
    #'gauss'     Gaussian-distributed additive noise.
    #'s&p'       Replaces random pixels with 0 or 1.
    #'speckle'   Multiplicative noise using out = image + n*image,where
    #        n is uniform noise with specified mean & variance.
    #       
    #'rotate'    rotates the image by a given angle
    #'seg-fel'   operates a felzenszwalb segmentation on the image
    #'seg -swirl'operates a swirl segmentation on the image.
    
   if noise_typ == "gauss":
      row,col, ch = image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
  
   elif noise_typ == "s&p":
      row,col = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size()[0] * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1
      # Pepper mode
      num_pepper = np.ceil(amount* image.size()[0] * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out

   elif noise_typ =="speckle":
      row,col = image.shape
      gauss = np.random.randn(row,col)
      gauss = gauss.reshape(row,col)        
      noisy = image + image * gauss
      return noisy
   
   elif noise_typ == "rotate":
       noisy = rotate(image, rotation)
       return noisy
   
   elif noise_typ == "seg-fel":
       noisy = seg.felzenszwalb(image)
       return noisy
   
   elif noise_typ == "seg-swirl": 
       noisy = swirl(image, rotation=0, strength=5, radius=120)
       return noisy
   
    
def create_z(orig_data, noise_list = ['gauss','seg-swirl'], limit = True):
    z = []
    for index, image in enumerate(orig_data):
        if limit:
            if index>19:
                break
        temp_image = image[0]
        for noise in noise_list:
            temp_image = (noisy(noise,temp_image))
        z.append(temp_image)

    return z 
'''