#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:44:27 2020

@author: michael
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

class statistical_estimator_big(nn.Module):
    def __init__(self, input_size = 1, output_size = 10):
        super().__init__()
        # Define the networks steps:
        self.conv1 = nn.Conv2d(input_size, 16, 3, stride = 1, padding = 1)
        self.batchNorm1 = nn.BatchNorm2d(16, momentum=1)
        self.relu = nn.ReLU(inplace = True)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride = 1, padding = 1)
        self.batchNorm2 = nn.BatchNorm2d(32, momentum=1)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride = 1, padding = 1)
        self.batchNorm3 = nn.BatchNorm2d(64, momentum=1)
        self.pool3 = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(64, 128, 3, stride = 1, padding = 1)
        self.batchNorm4 = nn.BatchNorm2d(128, momentum=1)
        self.pool4 = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(64,output_size)

        #self.softmax = nn.Softmax(dim = 0)
    
    def forward(self, x):
        #x = torch.from_numpy(x)
        x = self.pool1(self.relu(self.batchNorm1(self.conv1(x))))
        x = self.pool2(self.relu(self.batchNorm2(self.conv2(x))))
        x = self.pool3(self.relu(self.batchNorm3(self.conv3(x))))
        x = self.pool4(self.relu(self.batchNorm4(self.conv4(x))))
        x = x.view(x.size(0), 64)
        x = self.fc(x)
        #x = self.softmax(x)
        
        return x
        
class statistical_estimator_small(nn.Module):
    def __init__(self, input_size = 1, output_size = 10):
        super().__init__()
        # Define the networks steps:
        self.conv1 = nn.Conv2d(input_size, 32, 3, stride = 1, padding = 1)
        self.batchNorm1 = nn.BatchNorm2d(32, momentum=1)
        self.relu = nn.ReLU(inplace = True)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride = 1, padding = 1)
        self.batchNorm2 = nn.BatchNorm2d(64, momentum=1)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(64,output_size)
    
    def forward(self, x):
        #x = torch.from_numpy(x)
        x = self.pool1(self.relu(self.batchNorm1(self.conv1(x))))
        x = self.pool2(self.relu(self.batchNorm2(self.conv2(x))))
        x = x.view(x.size(0), 64*7*7)
        x = self.fc(x)
        #x = self.softmax(x)
        
        return x
    
class statistical_estimator_DCGAN(nn.Module):
    def __init__(self, input_size = 2, output_size = 1):
        super().__init__()
        # Define the networks steps:
        self.conv1 = nn.Conv2d(input_size, 16, 5, stride = 2, padding = 2)
        self.elu = nn.ELU(inplace = True)
        self.conv2 = nn.Conv2d(16, 32, 5, stride = 2, padding = 2)
        self.conv3 = nn.Conv2d(32, 64, 5, stride = 2, padding = 2)
        self.fc = nn.Linear(1024,output_size)
    
    def forward(self, x1,x2):
        x = torch.cat((x1.float().unsqueeze(1),x2.float().unsqueeze(1)),1)
        #x = torch.from_numpy(x)
        x = self.elu(self.conv1(x))
        x = self.elu(self.conv2(x))
        x = self.elu(self.conv3(x))
        x = x.view(x.size(0), 64*4*4)
        x = self.fc(x)
        #x = self.softmax(x)
        
        return x
    
class statistical_estimator_DCGAN_2(nn.Module):
    def __init__(self, input_size = 1, output_size = 1):
        super().__init__()
        # Define the networks steps:
        self.conv1 = nn.Conv2d(input_size, 16, 5, stride = 2, padding = 2)
        self.elu = nn.ELU(inplace = True)
        self.conv2 = nn.Conv2d(16, 32, 5, stride = 2, padding = 2)
        self.conv3 = nn.Conv2d(64, 64, 5, stride = 2, padding = 2)
        self.fc = nn.Linear(1024,output_size)
        # Define the networks steps:
        self.conv12 = nn.Conv2d(input_size, 16, 5, stride = 2, padding = 2)
        self.elu = nn.ELU(inplace = True)
        self.conv22 = nn.Conv2d(16, 32, 5, stride = 2, padding = 2)
        self.fc = nn.Linear(1024,output_size)
    
    def forward(self, x1,x2):
        x1 = x1.float().unsqueeze(1)
        x2 = x2.float().unsqueeze(1)
        x1 = self.elu(self.conv1(x1))
        x1 = self.elu(self.conv2(x1))
        x2 = self.elu(self.conv12(x2))
        x2 = self.elu(self.conv22(x2))
        x = torch.cat((x1.float(),x2.float()),1)
        x = self.elu(self.conv3(x))
        x = x.view(x.size(0), 64*4*4)
        x = self.fc(x)
        
        return x
    
class statistical_estimator_DCGAN_3(nn.Module):
    def __init__(self, input_size = 1, output_size = 1,number_descending_blocks=4, number_repeating_blocks = 3, repeating_blockd_size = 512):
        super().__init__()
        if type(repeating_blockd_size) == int:
            repeating_blockd_size = [repeating_blockd_size]
        # Define the networks steps:
        self.conv1 = nn.Conv2d(input_size, 16, 5, stride = 2, padding = 2)
        self.elu = nn.ELU(inplace = True)
        self.conv2 = nn.Conv2d(16, 32, 5, stride = 2, padding = 2)
        self.conv3 = nn.Conv2d(32, 64, 5, stride = 2, padding = 2)
        #self.fc = nn.Linear(1024,output_size)
        # Define the networks steps:
        self.conv12 = nn.Conv2d(input_size, 16, 5, stride = 2, padding = 2)
        self.elu = nn.ELU(inplace = True)
        self.conv22 = nn.Conv2d(16, 32, 5, stride = 2, padding = 2)
        self.conv32 = nn.Conv2d(32, 64, 5, stride = 2, padding = 2)
        fc = []
        in_dim = int(2048)
        if number_descending_blocks>7:
            print('number_descending_blocks should be <= 7, assigning descending_block_size = 7')
            number_descending_blocks = 7
        for i in range(number_descending_blocks):
            new_dim = int(in_dim/2)
            if in_dim in repeating_blockd_size:
                for j in range(number_repeating_blocks):
                    fc.append(nn.Linear(in_dim,in_dim))
            else:
                fc.append(nn.Linear(in_dim,new_dim))
            in_dim = new_dim
            
        self.fc = nn.ModuleList(fc)
        #self.fc12 = nn.Linear(1024,256)
        self.fc_last = nn.Linear(fc[-1].out_features,output_size)
    
    def forward(self, x1,x2):
        x1 = x1.float().unsqueeze(1)
        x2 = x2.float().unsqueeze(1)
        x1 = self.elu(self.conv1(x1))
        x1 = self.elu(self.conv2(x1))
        x1 = self.elu(self.conv3(x1))
        x2 = self.elu(self.conv12(x2))
        x2 = self.elu(self.conv22(x2)) 
        x2 = self.elu(self.conv32(x2))
        x1 = x1.view(x1.size(0), 64*4*4)
        x2 = x2.view(x2.size(0), 64*4*4)
        x = torch.cat((x1,x2),1) 
        for idx, des_fc in enumerate(self.fc):
            x = self.elu(des_fc(x))
        x = self.fc_last(x)
        
        return x
    
class statistical_estimator_syclope(nn.Module):
    def __init__(self, input_size = 1, output_size = 1):
        super().__init__()
        # Define the networks steps:
        self.actions_conv1 = nn.Conv1d(9, 6, 3,stride = 2)
        self.actions_conv2 = nn.Conv1d(6, 4, 3, stride = 2)
        self.conv1 = nn.Conv2d(input_size, 16, 5, stride = 2, padding = 2)
        self.elu = nn.ELU(inplace = True)
        self.conv2 = nn.Conv2d(16, 32, 5, stride = 2, padding = 2)
        self.conv3 = nn.Conv2d(32, 64, 5, stride = 2, padding = 2)
        self.fc = nn.Linear(1024,output_size)
        
        
    def forward(self, x1,x2):
        x1 = x1.float().unsqueeze(0).unsqueeze(0)
        print(x1.size())
        x2 = x2.float().unsqueeze(0).unsqueeze(0)
        x2 = self.elu(self.actions_conv1(x2))
        x2 = self.elu(self.actions_conv2(x2))
        x2 = x2.view(x2.size(0),249*4)
        print(x2.size())
        #x = torch.from_numpy(x)
        x1 = self.elu(self.conv1(x1))
        x1 = self.elu(self.conv2(x1))
        x1 = self.elu(self.conv3(x1))
        x1 = x1.view(x1.size(0), 64*4*4)
        print(x1.size())
        x = torch.cat((x1,x2),1)
        print(x.size())
        x1 = self.fc(x1)
        #x = self.softmax(x)
        
        return x1
    
class trajectory_classifier(nn.Module):
    def __init__(self, input_size = 1, output_size = 1):
        super().__init__()
        self.actions_conv1 = nn.Conv1d(9,18,3)
        self.batchNorm1 = nn.BatchNorm1d(18)
        self.actions_conv2 = nn.Conv1d(18,36,5)
        self.batchNorm2 = nn.BatchNorm1d(36)
        self.actions_conv3 = nn.Conv1d(36,9,3)
        self.batchNorm3 = nn.BatchNorm1d(9)
        self.fc1 = nn.Linear(122*9, 512)
        self.fc2 = nn.Linear(512, 10)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x = self.pool(self.relu(self.batchNorm1(self.actions_conv1(x))))
        x = self.pool(self.relu(self.batchNorm2(self.actions_conv2(x))))
        x = self.pool(self.relu(self.batchNorm3(self.actions_conv3(x))))
        x = x.view(x.size(0), 122*9)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
       
        return x
        