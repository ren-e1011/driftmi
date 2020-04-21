#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 14:44:27 2020

@author: michael
"""
import torch
import torch.optim as optim
import pandas as pd
import MINE
import sys
import os

# sys.argv gets 4 values:
# [1] - the optimizer - one of three Adam, SGD, RMSprop
# [2] - lr
# [3] - batch_size
# [4] - epochs
# [5] - train true/false 1/0
# [6] - Net num - 1-3
# [7] - traject/MNIST/combined
# [8] - number_descending_blocks - i.e. how many steps should the fc networks take,
#                               starting at 2048 nodes and every step divide the size 
#                                by 2. max number is 7                                
# [9] - number_repeating_blocks - the number of times to repeat a particular layer
# [10] - repeating_blockd_size - the size of nodes of the layer to repeat
# [11] - traject_max_depth
# [12] - traject_num_layers
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


gamma = 0
optimizer = int(sys.argv[1])
lr = float(sys.argv[2])
batch_size = int(sys.argv[3])
epochs = int(sys.argv[4])
if int(sys.argv[5]) == 1:
    train = True
else:
    train = False
net_num = int(sys.argv[6])
traject = str(sys.argv[7])
number_descending_blocks = int(sys.argv[8])
number_repeating_blocks = int(sys.argv[9])
repeating_blockd_size = int(sys.argv[10])
traject_max_depth = int(sys.argv[11])
traject_num_layers = int(sys.argv[12])
mine = MINE.MINE(train = train, 
                 traject = traject, 
                 batch = batch_size, 
                 lr = lr,
                 gamma = gamma, 
                 optimizer = optimizer,
                 net_num = net_num, 
                 traject_max_depth = traject_max_depth, 
                 traject_num_layers = traject_num_layers, 
                 traject_stride = [3,1],
                 traject_kernel = 5, 
                 traject_padding = 0,
                 traject_pooling = [1,2], 
                 number_descending_blocks = number_descending_blocks, 
                 number_repeating_blocks=number_repeating_blocks,  
                 repeating_blockd_size = repeating_blockd_size)

print(mine.net)
safty = 0
print('Epochs: {}'.format(int(sys.argv[4])))
mine.train(epochs = int(sys.argv[4]))
while len(results) == 0:
    mine.restart_network()
    check_trial = mine.train(epochs = epochs)
    safty+=1
    if safty>5:
        break
safty = 0
results = pd.DataFrame(mine.results)
nm = 'results_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}'.format(sys.argv[1],int(float(sys.argv[2])*10000),sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8],sys.argv[9],sys.argv[10])
results.to_pickle(results, nm)
#results.to_csv('results_{0}_{1}_{2}_{3}_{4}.csv'.format(sys.argv[1],int(float(sys.argv[2])*10000),sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8]))
torch.save(mine.net.state_dict, 'net_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}'.format(sys.argv[1],int(float(sys.argv[2])*10000),sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8],sys.argv[9],sys.argv[10]))

