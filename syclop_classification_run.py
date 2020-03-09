#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 14:44:27 2020

@author: michael
"""

import pandas as pd
import MINE
import sys
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import pickle

import pyprind
import networks
import utils


# sys.argv gets 4 values:
# [1] - the optimizer - one of three Adam, SGD, RMSprop
# [2] - lr
# [3] - batch_size
# [4] - epochs
# [5] - Net num - 1-3
# [6] - Dropout
# [7] - max_depth
# [8] - number_ascending_blocks - i.e. how many steps should the conv networks take,
#                               starting at 9                                
# [9] - number_descending_blocks - i.e. how many steps should the fc networks take,
#                               starting at 2048 nodes and every step divide the size 
#                                by 2. max number is 7                                
# [10] - number_repeating_blocks - the number of times to repeat a particular layer
# [11] - repeating_blockd_size - the size of nodes of the layer to repeat
# [12] - data_path - if the data is stored outside the working directory. If not 
#                    sys.argv[0] should be 0
#For example, number_descending_blocks = 5, number_repeating_blocks = 3,
# and repeating_blockd_size = 512 then the net architecture will look like:
#statistical_estimator_DCGAN_3(
#  (conv1): Conv2d(1, 16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
#  (elu): ELU(alpha=1.0, inplace=True)
#  (conv2): Conv2d(16, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
#  (conv3): Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
#  (conv12): Conv2d(1, 16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
#  (conv22): Conv2d(16, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
#  (conv32): Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
#  (fc): ModuleList(
#    (0): Linear(in_features=2048, out_features=1024, bias=True)
#    (1): Linear(in_features=1024, out_features=512, bias=True)
#    (2): Linear(in_features=512, out_features=512, bias=True)
#    (3): Linear(in_features=512, out_features=512, bias=True)
#    (4): Linear(in_features=512, out_features=512, bias=True)
#    (5): Linear(in_features=256, out_features=128, bias=True)
#    (6): Linear(in_features=128, out_features=64, bias=True)
#  )
#  (fc_last): Linear(in_features=64, out_features=1, bias=True)
#)

def check_accuracy(dataset):
    net.eval()
    test_loss = 0 
    correct = 0
    test_losses = []
    with torch.no_grad():
        for data, target in dataset:
            output = net(data)
            test_loss += loss_func(output, target).item()
            pred = output.data.max(1, keepdim = True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        test_accuracy = 100.*correct/len(dataset.dataset)
        #print("test loss = {} , percentage of correct answers = {}".format(test_loss,100.*correct/len(dataset.dataset)))
    return test_accuracy, np.mean(test_losses)

#Transforming the sys.argv to training veriables
optimizer = int(sys.argv[1])
lr = float(sys.argv[2])
batch_size = int(sys.argv[3])
epochs = int(sys.argv[4])
net_num = int(sys.argv[5])
dropout = float(sys.argv[6])
max_depth = int(sys.argv[7])
number_ascending_blocks = int(sys.argv[8])
number_descending_blocks = int(sys.argv[9])
number_repeating_blocks = int(sys.argv[10])
repeating_blockd_size = int(sys.argv[11])
data_path = str(sys.argv[12]) #
#if data_path is 0 we use the cwd:
if len(data_path) == 1:
    data_path = os.getcwd()
train, test = utils.create_train_test(dr = data_path)
net = networks.trajectory_classifier_dropout(p=dropout, max_depth = max_depth)
#Defining the optimizer
if type(optimizer) != int: 
    optimizer = int(optimizer)
if optimizer == 1:
    optimizer = optim.SGD(net.parameters(), lr = lr)
    print('')
    print('Optimizer: SGD')
elif optimizer == 2:
    optimizer = optim.Adam(net.parameters(), lr = lr)
    print('')
    print('Optimizer: Adam')
else:
    optimizer = optim.RMSprop(net.parameters(), lr = lr)
    print('')
    print('Optimizer: SGD')

print('defined network')
optimizer = optim.Adam(net.parameters(), lr=3e-3)
loss_func = nn.CrossEntropyLoss()


bar = pyprind.ProgBar(len(train)/batch_size*epochs, monitor = True)
dataloader = torch.utils.data.DataLoader(train,  batch_size = batch_size, shuffle = True)
train_loss = []
test_loader = torch.utils.data.DataLoader(test,  batch_size = len(test), shuffle = True)
test_accuracies = []
train_accuracies = []
for i in range(epochs):
    test_accur, _ = check_accuracy(test_loader)
    test_accuracies.append(test_accur)
    net.train()
    #print('Starting Epoch Number {}'.format(i+1))
    batch_loss = []
    correct = 0
    for batch_idx, (data,target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = net(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        bar.update()
        
        batch_loss.append(loss.item())
        pred = output.data.max(1, keepdim = True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    if i%50==0:
        print('')
        print('Loss is {}'.format(np.mean(batch_loss)))
        print('Training Accuracy is:')
        train_accur, _ = check_accuracy(dataloader)
        print(train_accur)
        print('Test Accuracy is: {}'.format(test))
        
    train_accuracy = 100.*correct/len(dataloader.dataset)  
    train_accuracies.append(train_accuracy)
    train_loss.append(np.mean(batch_loss))
'''
print('Training Accuracy is:')
check_accuracy(dataloader)
plt.figure(figsize = [12,8])
plt.plot(train_loss)
plt.title('Mean Batch Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss - {}'.format(loss_func))

plt.figure(figsize = [12,8])
plt.plot(test_accuracies)
plt.title('Test Accuracies Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracies')
'''

name_test= 'test_accuracies_{0}_{1}_{2}_{3}_{4}_{5}_{6}'.format(sys.argv[1],int(float(sys.argv[2])*10000),sys.argv[3],sys.argv[4],sys.argv[5],int(float(sys.argv[6])*10),sys.argv[7])
name_train= 'train_accuracies_{0}_{1}_{2}_{3}_{4}_{5}_{6}'.format(sys.argv[1],int(float(sys.argv[2])*10000),sys.argv[3],sys.argv[4],sys.argv[5],int(float(sys.argv[6])*10),sys.argv[7])
name_train_loss= 'train_loss_{0}_{1}_{2}_{3}_{4}_{5}_{6}'.format(sys.argv[1],int(float(sys.argv[2])*10000),sys.argv[3],sys.argv[4],sys.argv[5],int(float(sys.argv[6])*10),sys.argv[7])
pickle.dump( test_accuracies, open( name_test, "wb" ) )
pickle.dump( train_accuracies, open( name_train, "wb" ) )
pickle.dump( train_loss, open( name_train_loss, "wb" ) )
#results.to_csv('results_{0}_{1}_{2}_{3}_{4}.csv'.format(sys.argv[1],int(float(sys.argv[2])*10000),sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8]))
torch.save(net.state_dict, 'net_{0}_{1}_{2}_{3}_{4}_{5}_{6}'.format(sys.argv[1],int(float(sys.argv[2])*10000),sys.argv[3],sys.argv[4],sys.argv[5],int(float(sys.argv[6])*10),sys.argv[7]))

