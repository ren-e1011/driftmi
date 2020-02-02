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
# [0] - the optimizer - one of three Adam, SGD, RMSprop
# [1] - lr
# [2] - batch_size
# [3] - epochs
# [4] - train true/false 1/0



if sys.argv[5] == 1:
    train = True
else:
    train = False
    
print('Train mode: {}'.format(train))
lr = float(sys.argv[2])
print('Learning Rate: {}'.format(lr))
batch_size = int(sys.argv[3])
print('Batch Size: {}'.format(batch_size))
mine = MINE.MINE(train = train, batch = batch_size, lr = lr)
if sys.argv[1] == 1:
    mine.mine_net_optim = optim.Adam(mine.net.parameters(), lr = lr)
    print('Optimizer: Adam')
elif sys.argv[1] == 2:
    mine.mine_net_optim = optim.SGD(mine.net.parameters(), lr = lr)
    print('Optimizer: SGD')
else:
    mine.mine_net_optim = optim.RMSprop(mine.net.parameters(), lr = lr)
    print('Optimizer: RMSprop')

print('Epochs: {}'.format(int(sys.argv[4])))
results = mine.train(epochs = int(sys.argv[4]))
results = pd.DataFrame(results)
nm = 'results_{0}_{1}_{2}_{3}_{4}'.format(sys.argv[1],int(float(sys.argv[2])*10000),sys.argv[3],sys.argv[4],sys.argv[5])
results.to_pickle(nm)
results.to_csv('results_{0}_{1}_{2}_{3}_{4}.csv'.format(sys.argv[1],int(float(sys.argv[2])*10000),sys.argv[3],sys.argv[4],sys.argv[5]))
torch.save(mine.net.state_dict, 'net_{0}_{1}_{2}_{3}_{4}'.format(sys.argv[1],int(float(sys.argv[2])*10000),sys.argv[3],sys.argv[4],sys.argv[5]))

