


import torch
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import MINE
import networks

trial = MINE.MINE(train = True, batch = 1000)

results_SGD = []
print('Running SGD trial')
batch_size = [1000,4000,8000]
lr_list = [5e-4,1e-3,3e-3]
count = 1
fig = plt.figure(figsize=[12,20])
fig.subplots_adjust(hspace=1, wspace=0.4)
for batch_s in batch_size:
    trial = MINE.MINE(train = True, batch = batch_s)
    for lr in lr_list:
        for i in range(2):
            ax = fig.add_subplot(3, 1, count)
            trial.net = networks.statistical_estimator_DCGAN(input_size = 2, output_size = 1)
            trial.mine_net_optim = optim.SGD(trial.net.parameters(), lr = lr)
            
            check_trial = trial.train(epochs = 80)    
            results_SGD.append(check_trial)
            torch.save(trial.net.state_dict, '/home/michael/Documents/Scxript/Syclop-MINE-master/trained networks/net_SGD_{0}_{1}'.format(batch_s,int(lr*10000)+i))
            print(check_trial[-1])
            ax = fig.add_subplot(3,1,count)
            ax.plot(check_trial[:],label = 'lr={}'.format(lr) )
            
    plt.legend(loc=0, frameon=False)
    plt.title('Mutual Information Neural Estimator MNIST - MNIST batch_size={}'.format(batch_s))
    plt.xlabel('Epochs')
    plt.ylabel('bits')
    count+=1

results_SGD = pd.DataFrame(results_SGD)
results_SGD.to_pickle('results_SGD')
results_SGD.to_csv('results_SGD.csv')

trial = MINE.MINE(train = True, batch = 1000)

results_Adam = []
print('Running Adam trial')
batch_size = [1000,4000, 8000]
lr_list = [5e-4,1e-3,3e-3]
count = 1
fig = plt.figure()
fig = plt.figure(figsize=[12,20])
fig.subplots_adjust(hspace=1, wspace=0.4)
for batch_s in batch_size:
    trial = MINE.MINE(train = True, batch = batch_s)
    for lr in lr_list:
        for i in range(2):
            ax = fig.add_subplot(3, 1, count)
            trial.net = networks.statistical_estimator_DCGAN(input_size = 2, output_size = 1)
            trial.mine_net_optim = optim.Adam(trial.net.parameters(), lr = lr)
            
            check_trial = trial.train(epochs = 80)    
            results_Adam.append(check_trial)
            torch.save(trial.net.state_dict, '/home/michael/Documents/Scxript/Syclop-MINE-master/trained networks/net_Adam_{0}_{1}'.format(batch_s,int(lr*10000)+i))
            print(check_trial[-1])
            ax = fig.add_subplot(3,1,count)
            ax.plot(check_trial[:],label = 'lr={}'.format(lr) )
            
    plt.legend(loc=0, frameon=False)
    plt.title('Mutual Information Neural Estimator MNIST - MNIST batch_size={}'.format(batch_s))
    plt.xlabel('Epochs')
    plt.ylabel('bits')
    count+=1

results_Adam = pd.DataFrame(results_Adam)
results_Adam.to_pickle('results_Adam')
results_Adam.to_csv('results_SAdam.csv')

