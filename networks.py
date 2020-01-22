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
    def __init__(self, input_size = 1, classifier = False):
        super().__init__()
        # Define the networks steps:
        self.conv1 = nn.Conv2d(1, 16, 3, stride = 1, padding = 1)
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
        if classifier:
            self.fc = nn.Linear(64,10)
        else:
            self.fc = nn.Linear(64,1)
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
    def __init__(self, input_size = 1, classifier = False):
        super().__init__()
        # Define the networks steps:
        self.conv1 = nn.Conv2d(1, 32, 3, stride = 1, padding = 1)
        self.batchNorm1 = nn.BatchNorm2d(32, momentum=1)
        self.relu = nn.ReLU(inplace = True)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride = 1, padding = 1)
        self.batchNorm2 = nn.BatchNorm2d(64, momentum=1)
        self.pool2 = nn.MaxPool2d(2,2)
        if classifier:
            self.fc = nn.Linear(64*7*7,10)
        else:
            self.fc = nn.Linear(64*7*7,1)
        #self.softmax = nn.Softmax(dim = 0)
    
    def forward(self, x):
        #x = torch.from_numpy(x)
        x = self.pool1(self.relu(self.batchNorm1(self.conv1(x))))
        x = self.pool2(self.relu(self.batchNorm2(self.conv2(x))))
        x = x.view(x.size(0), 64*7*7)
        x = self.fc(x)
        #x = self.softmax(x)
        
        return x
        