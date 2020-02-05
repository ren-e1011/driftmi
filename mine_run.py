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
# [6] - gamma



if int(sys.argv[5]) == 1:
    train = True
else:
    train = False
#gamma = float(sys.argv[6])
lr = float(sys.argv[2])
batch_size = int(sys.argv[3])
optimizer = int(sys.argv[1])
mine = MINE.MINE(train = train, batch = batch_size, lr = lr, gamma = gamma, optimizer = optimizer)
safty = 0
print('Epochs: {}'.format(int(sys.argv[4])))
results = mine.train(epochs = int(sys.argv[4]))
while len(results) == 0:
    mine.restart_network()
    check_trial = mine.train(epochs = int(sys.argv[4]))
    safty+=1
    if safty>5:
        break
safty = 0
results = pd.DataFrame(results)
nm = 'results_{0}_{1}_{2}_{3}_{4}'.format(sys.argv[1],int(float(sys.argv[2])*10000),sys.argv[3],sys.argv[4],sys.argv[5])
results.to_pickle(nm)
results.to_csv('results_{0}_{1}_{2}_{3}_{4}.csv'.format(sys.argv[1],int(float(sys.argv[2])*10000),sys.argv[3],sys.argv[4],sys.argv[5]))
torch.save(mine.net.state_dict, 'net_{0}_{1}_{2}_{3}_{4}'.format(sys.argv[1],int(float(sys.argv[2])*10000),sys.argv[3],sys.argv[4]))

