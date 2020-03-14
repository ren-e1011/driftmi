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
import numpy as np

import utils

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
    def __init__(self, input_size = 1, output_size = 10,max_depth = 64):
        super().__init__()
        conv_depth_array = np.linspace(9,max_depth, 6).astype(int)
        self.actions_conv1 = nn.Conv1d(9,conv_depth_array[1],5, stride = 3)
        self.batchNorm1 = nn.BatchNorm1d(conv_depth_array[1])
        self.actions_conv2 = nn.Conv1d(conv_depth_array[1],conv_depth_array[2],5, stride = 3)
        
        self.batchNorm2 = nn.BatchNorm1d(conv_depth_array[2])
        self.actions_conv3 = nn.Conv1d(conv_depth_array[2],conv_depth_array[3],5, stride = 3)
        self.batchNorm3 = nn.BatchNorm1d(conv_depth_array[3])
        self.actions_conv4 = nn.Conv1d(conv_depth_array[3],conv_depth_array[4],5, stride = 3, padding = 4)
        self.batchNorm4 = nn.BatchNorm1d(conv_depth_array[4])
        self.actions_conv5 = nn.Conv1d(conv_depth_array[4],conv_depth_array[5],5, stride = 3, padding = 4)
        self.batchNorm5 = nn.BatchNorm1d(conv_depth_array[5])
        self.conv_depth_array = conv_depth_array
        self.fc1 = nn.Linear(3*self.conv_depth_array[5], 512)
        self.batchnormfc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 10)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x = self.pool(self.relu(self.batchNorm1(self.actions_conv1(x))))
        x = self.pool(self.relu(self.batchNorm2(self.actions_conv2(x))))
        x = self.relu(self.batchNorm3(self.actions_conv3(x)))
        x = self.relu(self.batchNorm4(self.actions_conv4(x)))
        x = self.relu(self.batchNorm5(self.actions_conv5(x)))
        x = x.view(x.size(0), 3*self.conv_depth_array[5])
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
       
        return x
    
class trajectory_classifier_dropout_fc(nn.Module):
    def __init__(self, input_size = 1, output_size = 10, p_fc = 0.5, max_depth = 64):
        super().__init__()
        conv_depth_array = np.linspace(9,max_depth, 6).astype(int)
        self.actions_conv1 = nn.Conv1d(9,conv_depth_array[1],5, stride = 3)
        self.batchNorm1 = nn.BatchNorm1d(conv_depth_array[1])
        self.actions_conv2 = nn.Conv1d(conv_depth_array[1],conv_depth_array[2],5, stride = 3)
        
        self.batchNorm2 = nn.BatchNorm1d(conv_depth_array[2])
        self.actions_conv3 = nn.Conv1d(conv_depth_array[2],conv_depth_array[3],5, stride = 3)
        self.batchNorm3 = nn.BatchNorm1d(conv_depth_array[3])
        self.actions_conv4 = nn.Conv1d(conv_depth_array[3],conv_depth_array[4],5, stride = 3, padding = 4)
        self.batchNorm4 = nn.BatchNorm1d(conv_depth_array[4])
        self.actions_conv5 = nn.Conv1d(conv_depth_array[4],conv_depth_array[5],5, stride = 3, padding = 4)
        self.batchNorm5 = nn.BatchNorm1d(conv_depth_array[5])
        self.conv_depth_array = conv_depth_array
        self.fc1 = nn.Linear(3*self.conv_depth_array[5], 3*self.conv_depth_array[5])
        self.dropfc1 = nn.Dropout(p=p_fc)
        self.fc2 = nn.Linear(3*self.conv_depth_array[5], 512)
        self.dropfc2 = nn.Dropout(p=p_fc)
        self.fc3 = nn.Linear(512, output_size)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x = self.pool(self.relu(self.batchNorm1(self.actions_conv1(x))))
        x = self.pool(self.relu(self.batchNorm2(self.actions_conv2(x))))
        x = self.relu(self.batchNorm3(self.actions_conv3(x)))
        x = self.relu(self.batchNorm4(self.actions_conv4(x)))
        x = self.relu(self.batchNorm5(self.actions_conv5(x)))
        x = x.view(x.size(0), 3*self.conv_depth_array[5])
        x = self.relu(self.dropfc1(self.fc1(x)))
        x = self.relu(self.dropfc2(self.fc2(x)))
        x = self.fc3(x)
       
        return x
    
class trajectory_classifier_dropout(nn.Module):
    def __init__(self, output_size = 10, p_conv=0.1, p_fc = 0.5, max_depth = 64, repeating_block = 2):
        super().__init__()
        conv_depth_array = np.linspace(9,max_depth, 5).astype(int)
        self.actions_conv1 = nn.Conv1d(9,conv_depth_array[1],5, stride = 3)
        self.drop1 = nn.Dropout(p=p_conv)
        self.actions_conv2 = nn.Conv1d(conv_depth_array[1],conv_depth_array[2],5, stride = 3)
        self.drop2 = nn.Dropout(p=p_conv)
        self.actions_conv3 = nn.Conv1d(conv_depth_array[2],conv_depth_array[3],5, stride = 3)
        self.drop3 = nn.Dropout(p=p_conv)
        self.actions_conv4 = nn.Conv1d(conv_depth_array[3],conv_depth_array[4],5, stride = 3, padding = 9)
        self.drop4 = nn.Dropout(p=p_conv)
        conv_block = []
        conv_drop = []
        for i in range(repeating_block):
            conv_block.append(nn.Conv1d(conv_depth_array[4],conv_depth_array[4],5, stride = 3, padding = 9))
            conv_drop.append(nn.Dropout(p=p_conv))
        self.conv_block = nn.ModuleList(conv_block)
        self.conv_drop = nn.ModuleList(conv_drop)
        #self.actions_conv5 = nn.Conv1d(conv_depth_array[4],conv_depth_array[4],5, stride = 3, padding = 9)
        #self.drop5 = nn.Dropout(p=p_conv)
        #self.actions_conv6 = nn.Conv1d(conv_depth_array[4],conv_depth_array[4],5, stride = 3, padding = 9)
        #self.drop6 = nn.Dropout(p=p_conv)
        self.actions_conv7 = nn.Conv1d(conv_depth_array[4],conv_depth_array[2],5, stride = 3, padding = 0)
        self.drop7 = nn.Dropout(p=p_conv)

        self.fc1 = nn.Linear(2*conv_depth_array[2], 512)
        self.dropfc1 = nn.Dropout(p=p_fc)
        self.fc2 = nn.Linear(512, output_size)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.conv_depth_array = conv_depth_array
    def forward(self,x):
        x = self.pool(self.relu(self.drop1(self.actions_conv1(x))))
        x = self.pool(self.relu(self.drop2(self.actions_conv2(x))))
        x = self.relu(self.drop3(self.actions_conv3(x)))
        x = self.relu(self.drop4(self.actions_conv4(x)))
        for idx, conv_layer in enumerate(self.conv_block):
            drop = self.conv_drop[idx]
            x = self.relu(drop(conv_layer(x)))
        #x = self.relu(self.drop5(self.actions_conv5(x)))
        #x = self.relu(self.drop6(self.actions_conv6(x)))
        x = self.relu(self.drop7(self.actions_conv7(x)))
        x = x.view(x.size(0), 2*self.conv_depth_array[2])
        x = self.relu(self.dropfc1(self.fc1(x)))
        x = self.fc2(x)
       
        return x
        
class trajectory_classifier_dropout2(nn.Module):
    def __init__(self, output_size = 10, p_conv=0.1, p_fc = 0.5, max_depth = 64):
        super().__init__()
        conv_depth_array = np.linspace(9,max_depth, 6).astype(int)
        self.actions_conv1 = nn.Conv1d(9,conv_depth_array[1],5, stride = 3)
        self.drop1 = nn.Dropout(p=p_conv)
        self.actions_conv2 = nn.Conv1d(conv_depth_array[1],conv_depth_array[2],5, stride = 3, padding = 0)
        self.drop2 = nn.Dropout(p=p_conv)
        self.actions_conv3 = nn.Conv1d(conv_depth_array[2],conv_depth_array[3],5, stride = 3,padding = 0)
        self.drop3 = nn.Dropout(p=p_conv)
        self.actions_conv4 = nn.Conv1d(conv_depth_array[3],conv_depth_array[4],5, stride = 3, padding = 0)
        self.drop4 = nn.Dropout(p=p_conv)
        self.actions_conv5 = nn.Conv1d(conv_depth_array[4],conv_depth_array[5],5, stride = 3, padding = 2)
        self.drop5 = nn.Dropout(p=p_conv)
        self.fc1 = nn.Linear(4*conv_depth_array[5], 4*conv_depth_array[5])
        self.dropfc1 = nn.Dropout(p=p_fc)
        self.fc2 = nn.Linear(4*conv_depth_array[5], 4*conv_depth_array[5])
        self.dropfc2 = nn.Dropout(p=p_fc)
        self.fc3 = nn.Linear(4*conv_depth_array[5], output_size)
        self.pool = nn.MaxPool1d(1)
        self.relu = nn.ReLU()
        self.conv_depth_array = conv_depth_array
    def forward(self,x):
        x = self.pool(self.relu(self.drop1(self.actions_conv1(x))))
        x = self.pool(self.relu(self.drop2(self.actions_conv2(x))))
        x = self.relu(self.drop3(self.actions_conv3(x)))
        x = self.relu(self.drop4(self.actions_conv4(x)))
        x = self.relu(self.drop5(self.actions_conv5(x)))
        x = x.view(x.size(0), 4*self.conv_depth_array[5])
        x = self.relu(self.dropfc1(self.fc1(x)))
        x = self.relu(self.dropfc2(self.fc2(x)))
        x = self.fc3(x)
       
        return x
    
class conv1d_classifier_(nn.Module):
    def __init__(self,input_dim = [9,1000,0], output_size = 10, p_conv=0.1, p_fc = 0.5, 
                 max_depth = 64, num_layers = 5, repeating_block_depth = 5, 
                 repeating_block_size = 2,stride = 3, kernel = 3, padding = [0,0,0,4,4],
                 pooling = 1):
        super().__init__()
        '''
        flexible network that calculates the network dimentions to accomidate changes
        in all veriables.
        Same padding on all layers vs. same padding once the size is x
        
        Variables:
            input_dim - Input dimantions of the data [dimentions, dim_w, dim_h]
            output_size            - output_size of classifier
            p_conv                 - dropout rate of conv layer
            p_fc                   - dropout rate of FC layers
            max_depth              - max depth of conv layers
            num_layers             - number of layers in the conv part
            reapeating_block_depth - what number in num_layers should we repeat
            repeating_block_number - how many times to repeat
            stride                 - stride of conv layers, could be a list 
                                     size num_layers or int
            kernel                 - kernel of conv layers, could be a list 
                                     size num_layers or int
            padding                - padding of conv layers, could be a list 
                                     size num_layers or int
            padding_when_too_small - is size of conv reduced below 4 to start 
                                     same padding on all layers. defult - True
                                     if False, will stop at the last layer. 
            pooling                - a list of layers where to insert a pooling
                                     layer, if int will put on all layers. 
                                     defult == 1 i.e. no pooling
                                     
                                
            
        
        '''
        
        
        self.p_conv = p_conv
        self.p_fc = p_fc
        conv_depth_array = np.linspace(input_dim[0],max_depth, num_layers+1).astype(int)
        if type(stride) == int:
            stride = np.ones(num_layers)*stride
            stride = stride.astype(int).tolist()
        else:
            if len(stride) < num_layers + repeating_block_size:
                #print('stride should be an integar or a num_layers list taking first element only')
                self.stride = np.ones(num_layers + repeating_block_size)*stride[-1]
                self.stride = self.stride.astype(int).tolist()
                for i in range(len(stride)):
                    self.stride[i] = stride[i]
        if type(kernel) == int:
            kernel = np.ones(num_layers)*kernel
            kernel = kernel.astype(int).tolist()
        else:
            if len(kernel) < num_layers:
                print('kernel should be an integar or a num_layers list taking first element only')  
                kernel = np.ones(num_layers)*kernel[0].astype(int).tolist()
        if type(padding) == int:
            padding = np.ones(num_layers)*padding
            padding = padding.astype(int).tolist()
        else:
            if len(padding) < num_layers:
                print('kernel should be an integar or a num_layers list taking first element only')  
                padding = np.ones(num_layers)*padding[0]
                padding = padding.astype(int).tolist()
        
        #define the pooling layers: 
        
        if type(pooling) == int:
            #Make sure pooling is not 0
            if pooling == 0:
                pooling == 1
            self.pooling_array = np.ones(num_layers+repeating_block_size)*pooling
            self.pooling_array = self.pooling_array.astype(int).tolist()
        else:
            #Make sure there is no 0 in the pooling arrays.
            if 0 in pooling:
                pooling[pooling.index(0)] = 1
            self.pooling_array = np.ones(num_layers+repeating_block_size)*pooling[-1]
            self.pooling_array = self.pooling_array.astype(int).tolist()
            for pool in range(len(pooling)):
                self.pooling_array[pool] = pooling[pool]
        
        #print(self.stride)       
        #Now to define the conv layers: 
        dim = []
        self.conv_layers = nn.ModuleList()
        dim_w = input_dim[1]
        for i in range(num_layers):
            dim_w_old = dim_w
            dim_w, _ = utils.conv1d_block_calculator(input_w = dim_w, kernel = kernel[i], stride = self.stride[i], padding = padding[i], pooling = self.pooling_array[i])
            #dim.append(dim_w)
            new_padding = 0
            #making sure the minimal size of dim_w is 4 or greater
            if dim_w < 4:
                while dim_w < 4:
                    new_padding += 1
                    dim_w, _ = utils.conv1d_block_calculator(input_w = dim_w_old, kernel = kernel[i], stride = self.stride[i], padding = new_padding, pooling = self.pooling_array[i])
                    
                self.conv_layers.append(nn.Conv1d(conv_depth_array[i],conv_depth_array[i+1],kernel[i], stride = self.stride[i], padding = new_padding))
            else:
                self.conv_layers.append(nn.Conv1d(conv_depth_array[i],conv_depth_array[i+1],kernel[i], stride = self.stride[i], padding = padding[i]))
             
            dim.append(dim_w)
            new_padding = 0
            if repeating_block_depth == 0:
                repeating_block_depth = False
            if i == repeating_block_depth - 1:
                
                dim_w_old = dim_w
                dim_w, _ = utils.conv1d_block_calculator(input_w = dim_w_old, kernel = kernel[i], stride = self.stride[i], padding = new_padding, pooling = self.pooling_array[i])
               
                while dim_w_old != dim_w:
                    
                    new_padding += 1
                    dim_w, _ = utils.conv1d_block_calculator(input_w = dim_w_old, kernel = kernel[i], stride = self.stride[i], padding = new_padding, pooling = self.pooling_array[i])

                #print(new_padding)
                for j in range(repeating_block_size):
                    self.conv_layers.append(nn.Conv1d(conv_depth_array[i+1],conv_depth_array[i+1],kernel[i], stride = self.stride[i], padding = new_padding))
                    
        #print(dim)
        self.final_conv_dim = dim_w.astype(int)
        self.fc_first_size = self.final_conv_dim*conv_depth_array[-1]

      
        #Creating a network with or without batchnorm/dropout
        if p_conv > 0:
            self.conv_drop = nn.ModuleList()
            for i in range(num_layers+repeating_block_size):
                self.conv_drop.append(nn.Dropout(p=p_conv))
        else:
            self.conv_batchNorm = nn.ModuleList()
            for i in range(num_layers):
                if i == repeating_block_depth:
                    for j in range(repeating_block_size):
                        self.conv_batchNorm.append(nn.BatchNorm1d(conv_depth_array[i+1]))
                else:
                    self.conv_batchNorm.append(nn.BatchNorm1d(conv_depth_array[i+1]))

        self.fc1 = nn.Linear(self.fc_first_size, self.fc_first_size)
        self.dropfc1 = nn.Dropout(p=p_fc)
        self.fc2 = nn.Linear(self.fc_first_size, self.fc_first_size)
        self.dropfc2 = nn.Dropout(p=p_fc)
        self.fc3 = nn.Linear(self.fc_first_size, output_size)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.conv_depth_array = conv_depth_array
    def forward(self,x):
        if self.p_conv:   
            for idx, conv_layer in enumerate(self.conv_layers):
                drop = self.conv_drop[idx]
                pool = nn.MaxPool1d(self.pooling_array[idx])
                x = pool(self.relu(drop(conv_layer(x))))
        else:
            for idx, conv_layer in enumerate(self.conv_layers):
                bNorm = self.conv_batchNorm[idx]
                pool = nn.MaxPool1d(self.pooling_array[idx])
                x = pool(self.relu(bNorm(conv_layer(x))))
        
        x = x.view(x.size(0), self.fc_first_size)
        x = self.relu(self.dropfc1(self.fc1(x)))
        x = self.relu(self.dropfc2(self.fc2(x)))
        x = self.fc3(x)
       
        return x
    
