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

class statistical_estimator_traject(nn.Module):
    def __init__(self, max_depth = 128):
        super().__init__()
        conv_depth_array = np.linspace(9,max_depth, 6).astype(int)
        self.actions_conv1 = nn.Conv1d(conv_depth_array[0],conv_depth_array[1],5, stride = 3)
        self.actions_conv2 = nn.Conv1d(conv_depth_array[1],conv_depth_array[2],5, stride = 1, padding = 0)
        self.actions_conv3 = nn.Conv1d(conv_depth_array[2],conv_depth_array[3],5, stride = 1,padding = 0)
        self.actions_conv4 = nn.Conv1d(conv_depth_array[3],conv_depth_array[4],5, stride = 1, padding = 0)
        self.actions_conv5 = nn.Conv1d(conv_depth_array[4],conv_depth_array[5],5, stride = 1, padding = 0)
        
        self.actions_conv1_2 = nn.Conv1d(conv_depth_array[0],conv_depth_array[1],5, stride = 3)
        self.actions_conv2_2= nn.Conv1d(conv_depth_array[1],conv_depth_array[2],5, stride = 1, padding = 0)
        self.actions_conv3_2 = nn.Conv1d(conv_depth_array[2],conv_depth_array[3],5, stride = 1,padding = 0)
        self.actions_conv4_2 = nn.Conv1d(conv_depth_array[3],conv_depth_array[4],5, stride = 1, padding = 0)
        self.actions_conv5_2 = nn.Conv1d(conv_depth_array[4],conv_depth_array[5],5, stride = 1, padding = 0)
        self.fc1 = nn.Linear(2*3*conv_depth_array[5], 3*conv_depth_array[5])
        self.fc2 = nn.Linear(3*conv_depth_array[5], conv_depth_array[5])
        self.fc3 = nn.Linear(conv_depth_array[5], 1)
        self.pool = nn.MaxPool1d(2)
        self.pool2 = nn.MaxPool1d(4)
        self.elu = nn.ELU()
        self.conv_depth_array = conv_depth_array
    def forward(self,x1,x2):
        
        x1 = self.elu(self.actions_conv1(x1))
        x1 = self.pool(self.elu(self.actions_conv2(x1)))
        x1 = self.pool(self.elu(self.actions_conv3(x1)))
        x1 = self.pool2(self.elu(self.actions_conv4(x1)))
        x1 = self.pool2(self.elu(self.actions_conv5(x1)))
        
        x1 = x1.view(x1.size(0), 3*self.conv_depth_array[5])
        x2 = self.elu(self.actions_conv1_2(x2))
        x2 = self.pool(self.elu(self.actions_conv2_2(x2)))
        x2 = self.pool(self.elu(self.actions_conv3_2(x2)))
        x2 = self.pool2(self.elu(self.actions_conv4_2(x2)))
        x2 = self.pool2(self.elu(self.actions_conv5_2(x2)))
        x2 = x2.view(x2.size(0), 3*self.conv_depth_array[5])
        x = torch.cat((x1,x2),1)
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.fc3(x)
       
        return x
    
class statistical_estimator(nn.Module):
    def __init__(self, traject_input_dim = [9,1000,0],p_conv=None, p_fc = None , 
                 traject_max_depth = 128, traject_num_layers = 3, traject_stride = [3,1],
                 traject_kernel = 3, traject_padding = 0,
                 traject_pooling = [1,2],number_descending_blocks=4, 
                 number_repeating_blocks = 3, repeating_blockd_size = 512, non_linearity = 'elu'):
            
        super().__init__()
        '''
        The statistical estimator for the MINE framework, recieve as input
        an image from MNIST dataset and a trajectory from the Syclop.
        Output - statistacl extimation of mutual information. 
        flexible network that calculates the network dimentions to accomidate changes
        in all veriables.
        The traject runs through Conv1d blocks with batchnorm for its architecture:
            traject_input_dim - Input dimantions of the data [dimentions, dim_w, dim_h]
            traject_max_depth       - max depth of conv layers
            traject_num_layers      - number of layers in the conv part
            traject_stride          - stride of conv layers, could be a list 
                                      or int if list size < network size
                                      will use the last value in the list for
                                      the rest
            traject_kernel          - kernel of conv layers, could be a list 
                                      or int if list size < network size
                                      will use the last value in the list for
                                      the rest
            traject_padding         - padding of conv layers, could be a list 
                                      or int if list size < network size
                                      will use the last value in the list for
                                      the rest
            padding_when_too_small  - is size of conv reduced below 4 to start 
                                      same padding on all layers. defult - True
                                      if False, will stop at the last layer. 
            traject_pooling         - a list of layers where to insert a pooling
                                      layer, if int will put on all layers. 
                                      defult == 1 i.e. no pooling
            
        The image runs through Conv2d blocks with batchnorm
        They are concated on the fc layers.
        
        Variables:
            
            output_size            - output_size of classifier
            p_conv                 - dropout rate of conv layer
            p_fc                   - dropout rate of FC layers
            
                                     
                                
            
        
        '''
        
        length_net = traject_num_layers
        if len(traject_stride) > length_net:
            traject_stride = traject_stride[:length_net]
        if len(traject_pooling) > length_net:
            traject_pooling = traject_pooling[:length_net]
            
        self.p_conv = p_conv
        self.p_fc = p_fc
        #conv_depth_array = np.linspace(traject_input_dim[0],traject_max_depth, traject_num_layers+1).astype(int)
        first_depth = 32
        conv_depth_array = [traject_input_dim[0],first_depth]
        for i in range(traject_num_layers):
            if conv_depth_array[-1]*2 > traject_max_depth:
                conv_depth_array.append(conv_depth_array[-1])
            else:
                conv_depth_array.append(conv_depth_array[-1]*2)
        if type(traject_stride) == int:
            self.stride = np.ones(traject_num_layers)*traject_stride
            self.stride = self.stride.astype(int).tolist()
        else:
            if len(traject_stride) < length_net:
                #print('stride should be an integar or a num_layers list taking first element only')
                self.stride = np.ones(length_net)*traject_stride[-1]
                self.stride = self.stride.astype(int).tolist()
                for i in range(len(traject_stride)):
                    self.stride[i] = traject_stride[i]
            else:
                self.stride = traject_stride
        if type(traject_kernel) == int:
            self.kernel = np.ones(length_net)*traject_kernel
            self.kernel = self.kernel.astype(int).tolist()
        else:
            if len(traject_kernel) < length_net:
                #print('stride should be an integar or a num_layers list taking first element only')
                self.kernel = np.ones(length_net)*traject_kernel[-1]
                self.kernel = self.kernel.astype(int).tolist()
                for i in range(len(traject_kernel)):
                    self.kernel[i] = traject_kernel[i]
            else:
                self.kernel = traject_kernel
                
        if type(traject_padding) == int:
            self.padding = np.ones(length_net)*traject_padding
            self.padding = self.padding.astype(int).tolist()
        else:
            if len(traject_kernel) < length_net:
                #print('stride should be an integar or a num_layers list taking first element only')
                self.padding = np.ones(length_net)*traject_padding[-1]
                self.padding = self.padding.astype(int).tolist()
                for i in range(len(traject_padding)):
                    self.padding[i] = traject_padding[i]
            else:
                self.padding = traject_padding
        #define the pooling layers: 
        if type(traject_pooling) == int:
            #Make sure pooling is not 0
            if traject_pooling == 0:
                traject_pooling == 1
            self.pooling_array = np.ones(length_net)*traject_pooling
            self.pooling_array = self.pooling_array.astype(int).tolist()
        else:
            #Make sure there is no 0 in the pooling arrays.
            while 0 in traject_pooling:
                traject_pooling[traject_pooling.index(0)] = 1
            self.pooling_array = np.ones(length_net)*traject_pooling[-1]
            self.pooling_array = self.pooling_array.astype(int).tolist()
            for pool in range(len(traject_pooling)):
                self.pooling_array[pool] = traject_pooling[pool]
        #Now to define the conv layers: 
        dim = []
        self.conv_layers = nn.ModuleList()
        dim_w = traject_input_dim[1]
        for i in range(traject_num_layers):
            dim_w_old = dim_w
            dim_w, _ = utils.conv1d_block_calculator(input_w = dim_w, kernel = self.kernel[i], stride = self.stride[i], padding = self.padding[i], pooling = self.pooling_array[i])
            #dim.append(dim_w)
            new_padding = self.padding[i]
            #making sure the minimal size of dim_w is 4 or greater
            if dim_w < 4:
                while dim_w < 4:
                    new_padding += 1
                    dim_w, _ = utils.conv1d_block_calculator(input_w = dim_w_old, kernel = self.kernel[i], stride = self.stride[i], padding = new_padding, pooling = self.pooling_array[i])
                    
                self.conv_layers.append(nn.Conv1d(conv_depth_array[i],conv_depth_array[i+1],self.kernel[i], stride = self.stride[i], padding = new_padding))
            else:
                self.conv_layers.append(nn.Conv1d(conv_depth_array[i],conv_depth_array[i+1],self.kernel[i], stride = self.stride[i], padding = self.padding[i]))
             
            dim.append(dim_w)
            new_padding = 0
                    
        #print(dim)
        self.final_conv_dim = dim_w.astype(int)
        self.fc_first_size = self.final_conv_dim*conv_depth_array[-1]

        if not p_conv:
            p_conv = 0
        #Creating a network with or without batchnorm/dropout
        if p_conv > 0:
            self.conv_drop = nn.ModuleList()
            for i in range(length_net):
                self.conv_drop.append(nn.Dropout(p=p_conv))
        else:
            self.conv_batchNorm = nn.ModuleList()
            for i in range(traject_num_layers):
                self.conv_batchNorm.append(nn.BatchNorm1d(conv_depth_array[i+1]))
                
        #Defining the Conv2d netwrok processing the image:
        if type(repeating_blockd_size) == int:
            repeating_blockd_size = [repeating_blockd_size]
        # Define the networks steps:
        self.conv1 = nn.Conv2d(1, 16, 5, stride = 2, padding = 2)
        self.conv2 = nn.Conv2d(16, 32, 5, stride = 2, padding = 2)
        self.conv3 = nn.Conv2d(32, 64, 5, stride = 2, padding = 2)
        self.non_linearity = non_linearity
        if self.non_linearity == 'elu':
            self.non_linearity = nn.ELU(inplace = True)
        elif self.non_linearity == 'leaky':
            self.non_linearity = nn.LeakyReLU()
        #Defining the joint fully connected layers:
        fc = []
        in_dim = int(1024 + self.fc_first_size)
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
        self.fc_last = nn.Linear(fc[-1].out_features,1)
                
    def forward(self,traject,image):
        if self.p_conv:   
            for idx, conv_layer in enumerate(self.conv_layers):
                drop = self.conv_drop[idx]
                pool = nn.MaxPool1d(self.pooling_array[idx])
                traject = pool(self.non_linearity(drop(conv_layer(traject))))
        else:
            for idx, conv_layer in enumerate(self.conv_layers):
                #bNorm = self.conv_batchNorm[idx]
                pool = nn.MaxPool1d(self.pooling_array[idx])
                traject = pool(self.non_linearity(conv_layer(traject)))
        traject = traject.view(traject.size(0), self.fc_first_size)
        
        image = image.float().unsqueeze(1)
        image = self.non_linearity(self.conv1(image))
        image = self.non_linearity(self.conv2(image))
        image = self.non_linearity(self.conv3(image))
        image = image.view(image.size(0), 64*4*4)
        
        x = torch.cat((image,traject),1) 
        for idx, des_fc in enumerate(self.fc):
            x = self.elu(des_fc(x))
        x = self.fc_last(x)
        
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
                fc.append(nn.Linear(in_dim,new_dim))
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
                 max_depth = 128, num_layers = 5, conv_depth_type = 'decending',repeating_block_depth = 5, 
                 repeating_block_size = 0,stride = [1], kernel = 3, padding = [0,0,0,4,4],
                 pooling = [4], BN = True):
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
        self.BN = BN
        length_net = num_layers+repeating_block_size
        if len(stride) > length_net:
            stride = stride[:length_net]
        if len(pooling) > length_net:
            pooling = pooling[:length_net]
            
        self.p_conv = p_conv
        self.p_fc = p_fc
        if conv_depth_type == 'decending':
            #conv_depth_array = np.linspace(input_dim[0],max_depth, num_layers+1).astype(int)
            first_depth = 32
            conv_depth_array = [input_dim[0],first_depth]
            
            for i in range(num_layers):
                if conv_depth_array[-1]*2 > max_depth:
                    conv_depth_array.append(conv_depth_array[-1])
                else:
                    conv_depth_array.append(conv_depth_array[-1]*2)
        elif conv_depth_type == 'same':
            conv_depth_array = max_depth*np.ones(num_layers+1).astype(int)
            conv_depth_array[0] = input_dim[0]
            
        
        if type(stride) == int:
            self.stride = np.ones(num_layers)*stride
            self.stride = self.stride.astype(int).tolist()
        else:
            if len(stride) < num_layers + repeating_block_size:
                #print('stride should be an integar or a num_layers list taking first element only')
                self.stride = np.ones(num_layers + repeating_block_size)*stride[-1]
                self.stride = self.stride.astype(int).tolist()
                for i in range(len(stride)):
                    self.stride[i] = stride[i]
            else:
                self.stride = stride
                
        if type(kernel) == int:
            kernel = np.ones(num_layers)*kernel
            self.kernel = kernel.astype(int).tolist()
        else:
            if len(kernel) < num_layers + repeating_block_size:
                #print('stride should be an integar or a num_layers list taking first element only')
                self.kernel = np.ones(num_layers + repeating_block_size)*kernel[-1]
                self.kernel = self.kernel.astype(int).tolist()
                for i in range(len(kernel)):
                    self.kernel[i] = kernel[i]
            else:
                self.kernel = kernel
        if type(padding) == int:
            padding = np.ones(num_layers)*padding
            self.padding = padding.astype(int).tolist()
        else:
            if len(padding) < num_layers + repeating_block_size:
                #print('stride should be an integar or a num_layers list taking first element only')
                self.padding = np.ones(num_layers + repeating_block_size)*padding[-1]
                self.padding = self.padding.astype(int).tolist()
                for i in range(len(padding)):
                    self.padding[i] = padding[i]
            else:
                self.padding = padding
        
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
            #print(kernel[i], self.stride[i], padding[i], self.pooling_array[i])
            dim_w, _ = utils.conv1d_block_calculator(input_w = dim_w, kernel = self.kernel[i], stride = self.stride[i], padding = self.padding[i], pooling = self.pooling_array[i])
            #dim.append(dim_w)
            new_padding = 0
            #making sure the minimal size of dim_w is 4 or greater
            if dim_w < 4:
                while dim_w < 4:
                    new_padding += 1
                    dim_w, _ = utils.conv1d_block_calculator(input_w = dim_w_old, kernel = self.kernel[i], stride = self.stride[i], padding = new_padding, pooling = self.pooling_array[i])
                    
                self.conv_layers.append(nn.Conv1d(conv_depth_array[i],conv_depth_array[i+1],self.kernel[i], stride = self.stride[i], padding = new_padding))
            else:
                self.conv_layers.append(nn.Conv1d(conv_depth_array[i],conv_depth_array[i+1],self.kernel[i], stride = self.stride[i], padding = self.padding[i]))
             
            dim.append(dim_w)
            new_padding = 0
            if repeating_block_depth == 0:
                repeating_block_depth = False
            if i == repeating_block_depth - 1:
                
                dim_w_old = dim_w
                dim_w, _ = utils.conv1d_block_calculator(input_w = dim_w_old, kernel = self.kernel[i], stride = self.stride[i], padding = new_padding, pooling = self.pooling_array[i])
               
                while dim_w_old != dim_w:
                    
                    new_padding += 1
                    dim_w, _ = utils.conv1d_block_calculator(input_w = dim_w_old, kernel = self.kernel[i], stride = self.stride[i], padding = new_padding, pooling = self.pooling_array[i])

                #print(new_padding)
                for j in range(repeating_block_size):
                    self.conv_layers.append(nn.Conv1d(conv_depth_array[i+1],conv_depth_array[i+1],self.kernel[i], stride = self.stride[i], padding = new_padding))
                    
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
        if self.fc_first_size > 1024:
            self.fc_second_size = 1024
        else:
            self.fc_second_size = self.fc_first_size
        self.fc2 = nn.Linear(self.fc_first_size, self.fc_second_size)
        self.dropfc2 = nn.Dropout(p=p_fc)
        self.fc3 = nn.Linear(self.fc_second_size, output_size)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.conv_depth_array = conv_depth_array
    def forward(self,x):
        if self.p_conv:   
            for idx, conv_layer in enumerate(self.conv_layers):
                drop = self.conv_drop[idx]
                pool = nn.MaxPool1d(self.pooling_array[idx])
                x = pool(self.relu(drop(conv_layer(x))))
        elif self.BN:
            for idx, conv_layer in enumerate(self.conv_layers):
                bNorm = self.conv_batchNorm[idx]
                pool = nn.MaxPool1d(self.pooling_array[idx])
                x = pool(self.relu(bNorm(conv_layer(x))))
        else:
            for idx, conv_layer in enumerate(self.conv_layers):
                pool = nn.MaxPool1d(self.pooling_array[idx])
                x = pool(self.relu(conv_layer(x)))        
        
        x = x.view(x.size(0), self.fc_first_size)
        x = self.relu(self.dropfc1(self.fc1(x)))
        x = self.relu(self.dropfc2(self.fc2(x)))
        x = self.fc3(x)
       
        return x
    
class conv1d_classifier_one_head(nn.Module):
    def __init__(self,input_dim = [9,1000,0], output_size = 10, p_conv=0.2, p_fc = 0.5, 
                 max_depth = 128, num_layers = 2, conv_depth_type = 'acending',repeating_block_depth = 5, 
                 repeating_block_size = 2,stride = [1], kernel = 3, padding = [0,0,0,4,4],
                 pooling = [4]):
        super().__init__()
        '''
        To be used a single head in a multi head classifier
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
        length_net = num_layers+repeating_block_size
        if len(stride) > length_net:
            stride = stride[:length_net]
        if len(pooling) > length_net:
            pooling = pooling[:length_net]
            
        self.p_conv = p_conv
        self.p_fc = p_fc
        if conv_depth_type == 'acending':
            #conv_depth_array = np.linspace(input_dim[0],max_depth, num_layers+1).astype(int)
            first_depth = 32
            conv_depth_array = [input_dim[0],first_depth]
            
            for i in range(num_layers):
                if conv_depth_array[-1]*2 > max_depth:
                    conv_depth_array.append(conv_depth_array[-1])
                else:
                    conv_depth_array.append(conv_depth_array[-1]*2)
                
        elif conv_depth_type == 'same':
            conv_depth_array = max_depth*np.ones(num_layers+1).astype(int)
            conv_depth_array[0] = input_dim[0]
            
        
        if type(stride) == int:
            self.stride = np.ones(num_layers)*stride
            self.stride = self.stride.astype(int).tolist()
        else:
            if len(stride) < num_layers + repeating_block_size:
                #print('stride should be an integar or a num_layers list taking first element only')
                self.stride = np.ones(num_layers + repeating_block_size)*stride[-1]
                self.stride = self.stride.astype(int).tolist()
                for i in range(len(stride)):
                    self.stride[i] = stride[i]
            else:
                self.stride = stride
                
        if type(kernel) == int:
            kernel = np.ones(num_layers)*kernel
            self.kernel = kernel.astype(int).tolist()
        else:
            if len(kernel) < num_layers + repeating_block_size:
                #print('stride should be an integar or a num_layers list taking first element only')
                self.kernel = np.ones(num_layers + repeating_block_size)*kernel[-1]
                self.kernel = self.kernel.astype(int).tolist()
                for i in range(len(kernel)):
                    self.kernel[i] = kernel[i]
            else:
                self.kernel = kernel
        if type(padding) == int:
            padding = np.ones(num_layers)*padding
            self.padding = padding.astype(int).tolist()
        else:
            if len(padding) < num_layers + repeating_block_size:
                #print('stride should be an integar or a num_layers list taking first element only')
                self.padding = np.ones(num_layers + repeating_block_size)*padding[-1]
                self.padding = self.padding.astype(int).tolist()
                for i in range(len(padding)):
                    self.padding[i] = padding[i]
            else:
                self.padding = padding
        
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
            #print(kernel[i], self.stride[i], padding[i], self.pooling_array[i])
            dim_w, _ = utils.conv1d_block_calculator(input_w = dim_w, kernel = self.kernel[i], stride = self.stride[i], padding = self.padding[i], pooling = self.pooling_array[i])
            #dim.append(dim_w)
            new_padding = 0
            #making sure the minimal size of dim_w is 4 or greater
            if dim_w < 4:
                while dim_w < 4:
                    new_padding += 1
                    dim_w, _ = utils.conv1d_block_calculator(input_w = dim_w_old, kernel = self.kernel[i], stride = self.stride[i], padding = new_padding, pooling = self.pooling_array[i])
                    
                self.conv_layers.append(nn.Conv1d(conv_depth_array[i],conv_depth_array[i+1],self.kernel[i], stride = self.stride[i], padding = new_padding))
            else:
                self.conv_layers.append(nn.Conv1d(conv_depth_array[i],conv_depth_array[i+1],self.kernel[i], stride = self.stride[i], padding = self.padding[i]))
             
            dim.append(dim_w)
            new_padding = 0
            if repeating_block_depth == 0:
                repeating_block_depth = False
            if i == repeating_block_depth - 1:
                
                dim_w_old = dim_w
                dim_w, _ = utils.conv1d_block_calculator(input_w = dim_w_old, kernel = self.kernel[i], stride = self.stride[i], padding = new_padding, pooling = self.pooling_array[i])
               
                while dim_w_old != dim_w:
                    
                    new_padding += 1
                    dim_w, _ = utils.conv1d_block_calculator(input_w = dim_w_old, kernel = self.kernel[i], stride = self.stride[i], padding = new_padding, pooling = self.pooling_array[i])

                #print(new_padding)
                for j in range(repeating_block_size):
                    self.conv_layers.append(nn.Conv1d(conv_depth_array[i+1],conv_depth_array[i+1],self.kernel[i], stride = self.stride[i], padding = new_padding))
                    
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
        if self.fc_first_size > 1024:
            self.fc_second_size = 1024
        else:
            self.fc_second_size = self.fc_first_size
        self.fc2 = nn.Linear(self.fc_first_size, self.fc_second_size)
        self.dropfc2 = nn.Dropout(p=p_fc)
        self.fc3 = nn.Linear(self.fc_second_size, output_size)
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
        
        return x

class conv1d_classifier_multi_head(nn.Module):
    def __init__(self, num_head = 3, kernel = [5,7,11]):
        '''
        A multi head trajectory classifier, combines num_head conv1d networks
        in a fc or conv layer to improve accuracy.
        
        '''
        super().__init__()
        if type(kernel)==int:
            self.kernel = kernel*np.ones(num_head).astype(int).tolist()
        elif len(kernel) < num_head:
            self.kernel = kernel[-1]*np.ones(num_head).astype(int).tolist()
            for ker in range(len(kernel)):
                self.kernel[ker] = kernel[ker]
        else:
            self.kernel = kernel
        self.heads = nn.ModuleList()
        self.size = []
        for i in range(num_head):
            temp_net = conv1d_classifier_one_head(kernel = self.kernel[i])
            self.size.append(temp_net.fc_first_size)
            self.heads.append(temp_net)
        
        self.first_fc_size = sum(self.size)
        self.fc1 = nn.Linear(self.first_fc_size, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 512)
        self.drop2 = nn.Dropout()
        self.fc3 = nn.Linear(512, 10)
        
        self.relu = nn.ReLU()
        
        
    def forward(self,x):
        x_list = []
        for idx, head in enumerate(self.heads):
            x_list.append(head(x))
        
        x = torch.cat(tuple(x_list), 1)
        x = self.relu(self.drop1(self.fc1(x)))
        x = self.relu(self.drop2(self.fc2(x)))
        x = self.fc3(x)
        
        return x
        
            
class trajectory_classifier_dropout_con2d(nn.Module):
    def __init__(self, output_size = 10, p_conv=0.1, p_fc = 0.5, max_depth = 64):
        super().__init__()
        conv_depth_array = np.linspace(1,max_depth, 6).astype(int)
        self.actions_conv1 = nn.Conv2d(1,conv_depth_array[1],[1,5], stride = [1,3])
        self.drop1 = nn.Dropout(p=p_conv)
        self.actions_conv2 = nn.Conv2d(conv_depth_array[1],conv_depth_array[2],[1,5], stride = 1, padding = 0)
        self.drop2 = nn.Dropout(p=p_conv)
        self.actions_conv3 = nn.Conv2d(conv_depth_array[2],conv_depth_array[3],[1,5], stride = 1,padding = 0)
        self.drop3 = nn.Dropout(p=p_conv)
        self.actions_conv4 = nn.Conv2d(conv_depth_array[3],conv_depth_array[4],[1,5], stride = 1, padding = 0)
        self.drop4 = nn.Dropout(p=p_conv)
        self.actions_conv5 = nn.Conv2d(conv_depth_array[4],conv_depth_array[5],[1,5], stride = 1, padding = 0)
        self.drop5 = nn.Dropout(p=p_conv)
        self.fc1 = nn.Linear(9*13*conv_depth_array[5], 4096)
        self.dropfc1 = nn.Dropout(p=p_fc)
        self.fc2 = nn.Linear(4096, 1024)
        self.dropfc2 = nn.Dropout(p=p_fc)
        self.fc3 = nn.Linear(1024, output_size)
        self.pool = nn.MaxPool2d(kernel_size=(1,2), stride = [1,2])
        self.relu = nn.ReLU()
        self.conv_depth_array = conv_depth_array
    def forward(self,x):
        x = x.unsqueeze(1)        
        x = self.pool(self.relu(self.drop1(self.actions_conv1(x))))        
        x = self.pool(self.relu(self.drop2(self.actions_conv2(x))))
        x = self.pool(self.relu(self.drop3(self.actions_conv3(x))))
        x = self.pool(self.relu(self.drop4(self.actions_conv4(x))))
        x = self.relu(self.drop5(self.actions_conv5(x)))
        x = x.view(x.size(0), 13*9*self.conv_depth_array[5])
        x = self.relu(self.dropfc1(self.fc1(x)))
        x = self.relu(self.dropfc2(self.fc2(x)))
        x = self.fc3(x)
       
        return x
    
class trajectory_classifier_simpel_regressor(nn.Module):
    def __init__(self, output_size = 10, p_fc = 0.5, number_descending_blocks=4, 
                 number_repeating_blocks = 3, repeating_blockd_size = 512):
        super().__init__()
        
        fc = []
        in_dim = int(1000)
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
        self.fc_last = nn.Linear(fc[-1].out_features,1)
        self.relu = nn.ReLU()
    def forward(self,x):
        for idx, des_fc in enumerate(self.fc):
            x = self.relu(des_fc(x))
        x = self.fc_last(x)
        
        return x
    
    
    
    
    
    
    
    
    
    
    
    