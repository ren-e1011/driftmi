import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import matplotlib.pyplot as plt
import os
import numpy as np

import pyprind

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
    
    