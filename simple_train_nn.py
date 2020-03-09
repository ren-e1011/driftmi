#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 10:00:35 2020

@author: michael
"""
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import os
import numpy as np

import pyprind
import networks
import utils

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
net = networks.trajectory_classifier_dropout2(p_conv = 0.2, p_fc = 0.5, max_depth = 512)
print('defined network')
optimizer = optim.Adam(net.parameters(), lr=3e-3)
loss_func = nn.CrossEntropyLoss()
epochs = 300
batch_size = 1000
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
    if i%5==0:
        print('')
        print('Epoch number {}'.format(i))
        print('Loss is {}'.format(np.mean(batch_loss)))
        print('Training Accuracy is:')
        train_accur, _ = check_accuracy(dataloader)
        print(train_accur)
        print('Test Accuracy is: {}'.format(test_accur))
    
    train_accuracy = 100.*correct/len(dataloader.dataset)  
    train_accuracies.append(train_accuracy)
    train_loss.append(np.mean(batch_loss))

print('Training Accuracy is:')
train_accur, _ = check_accuracy(dataloader)
print(train_accur)
plt.figure(figsize = [12,8])
plt.plot(train_accuracies)
plt.title('Train accuracies Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracies %')

plt.figure(figsize = [12,8])
plt.plot(test_accuracies)
plt.title('Test Accuracies Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracies')




    
#os.chdir('/home/michael/Documents/Scxript/Syclop-MINE-master/Trajectory_classification/')
#torch.save(net.state_dict, 'net_{}'.format(int(100.*correct/len(test_loader.dataset))))