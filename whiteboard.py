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

import pyprind
os.chdir('/home/michael/Documents/Scxript/')
from networks import statistical_estimator_big, statistical_estimator_small, statistical_estimator_DCGAN


mnist_data_train = torchvision.datasets.MNIST('home/michael/Documents/MNIST data', train = True, transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,),(0.3081,))]),download = True)

mnist_data_test = torchvision.datasets.MNIST('home/michael/Documents/MNIST data', train = False, transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,),(0.3081,))]), download = True)
    
train_loader = torch.utils.data.DataLoader(mnist_data_train, batch_size = 64, shuffle = True)
test_loader = torch.utils.data.DataLoader(mnist_data_test, batch_size = 1000, shuffle = True)


examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(example_data.shape)
'''
fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
fig
'''
net = statistical_estimator_DCGAN(input_size = 2, output_size = 10)
print('defined network')
optimizer = optim.Adam(net.parameters())
loss_func = nn.CrossEntropyLoss()
epochs = 1
bar = pyprind.ProgBar(len(train_loader)*epochs, monitor = True)
for i in range(epochs):
    for batch_idx, (data,target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = net(data, data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        bar.update()
    
net.eval()
test_loss = 0 
correct = 0
test_losses = []
with torch.no_grad():
    for data, target in test_loader:
        output = net(data,data)
        test_loss += loss_func(output, target).item()
        pred = output.data.max(1, keepdim = True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print("test loss = {} , percentage of correct answers = {}".format(test_loss, 
          100.*correct/len(test_loader.dataset)))
    

    