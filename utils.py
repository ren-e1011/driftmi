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

import numpy as np
import os
from skimage.transform import rescale, resize, downscale_local_mean, rotate
import skimage.segmentation as seg
from skimage.transform import swirl
import torch
import matplotlib.pyplot as plt 

class MNIST_for_MINE(Dataset):
    def __init__(self, train = False, transform = None):
        
        self.transform = transform
        
        for i in range(10):
            main_dataset = torchvision.datasets.MNIST('home/michael/Documents/MNIST data', 
                    train = train,download = True)
            sec_dataset = torchvision.datasets.MNIST('home/michael/Documents/MNIST data',
                    train = train, download = True)
                    
            
            idx = main_dataset.targets==i
            main_dataset.targets = main_dataset.targets[idx]
            main_dataset.data = main_dataset.data[idx]
            if i == 0:
                self.main_data = main_dataset.data
                self.main_targets = main_dataset.targets
            else:
                self.main_data = torch.utils.data.ConcatDataset([self.main_data, main_dataset.data])
                self.main_targets = torch.utils.data.ConcatDataset([self.main_targets, main_dataset.targets])
              
            idx = np.random.randint(0,len(sec_dataset),len(main_dataset))
            marginal_temp = torchvision.datasets.MNIST('home/michael/Documents/MNIST data', train = False, download = True)
            marginal_temp.targets = sec_dataset.targets[idx]
            marginal_temp.data = sec_dataset.data[idx]
            if i == 0:
                self.marginal_data = marginal_temp.data
                self.marginal_targets = marginal_temp.targets
            else:
                self.marginal_data = torch.utils.data.ConcatDataset([self.marginal_data, marginal_temp.data])
                self.marginal_targets = torch.utils.data.ConcatDataset([self.marginal_targets, marginal_temp.targets])
            
            
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