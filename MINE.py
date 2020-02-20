
'''

Using Mine to drew information from the 
MINE - Mutual Information Neural Estimator 

steps to take:
    1)drew two images for MINE
    2)create joint_disterbution and marginal distribution 
    3)repeat:
        
        

'''

import os
import numpy as np
import pandas as pd
import pyprind
#import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torchvision

import networks 
import utils



class MINE():
    def __init__(self, train = True, batch = 1000, lr = 3e-3, gamma = 0.001, optimizer=1, net_num = 1,
                 number_descending_blocks = 3, 
                 number_repeating_blocks=2, repeating_blockd_size=512):
        self.net_num = net_num
        if self.net_num == 1:
            self.net = networks.statistical_estimator_DCGAN(input_size = 2, output_size = 1)
        elif self.net_num == 2:
            self.net = networks.statistical_estimator_DCGAN_2(input_size = 1, output_size = 1)
        else:
            self.net = networks.statistical_estimator_DCGAN_3(input_size = 1, output_size = 1,number_descending_blocks = number_descending_blocks, 
                 number_repeating_blocks=number_repeating_blocks, repeating_blockd_size = repeating_blockd_size)
        self.number_descending_blocks = number_descending_blocks 
        self.number_repeating_blocks = number_repeating_blocks 
        self.repeating_blockd_size = repeating_blockd_size
        #self.input1 = input1
        #self.input2 = input2
        self.lr = lr
        self.optimizer = optimizer
        if type(optimizer) != int:
            optimizer = int(optimizer)
        if self.optimizer == 1:
            self.mine_net_optim = optim.SGD(self.net.parameters(), lr = self.lr)
            print('')
            print('Optimizer: SGD')
        elif self.optimizer == 2:
            self.mine_net_optim = optim.Adam(self.net.parameters(), lr = self.lr)
            print('')
            print('Optimizer: Adam')
        else:
            self.mine_net_optim = optim.RMSprop(self.net.parameters(), lr = self.lr)
            print('')
            print('Optimizer: SGD')
        
        
        
    
        self.dataset = utils.MNIST_for_MINE(train=train)
        self.train_value = train
        self.dataloader = torch.utils.data.DataLoader(self.dataset,  batch_size = batch, shuffle = True)    
        self.batch_size = batch
        
        #self.scheduler = optim.lr_scheduler.StepLR(self.mine_net_optim, step_size=10*(len(self.dataset)/self.batch_size), gamma=gamma)
        self.gamma = gamma
        self.scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(self.mine_net_optim, mode='max', factor=0.5, patience=10, verbose=False, threshold=0.0001, threshold_mode='abs', cooldown=0, min_lr=0, eps=1e-08)    
        
        
        #print all variables of the system:
        print('Learning rate = {}'.format(self.lr))
        print('Batch size = {}'.format(self.batch_size))
        #print('Gamma of lr decay = {}'.format(self.gamma))
        print('Using Net {}'.format(self.net_num))
        if self.net_num == 3:
            print('Number of Descending Blocks is {}'.format(self.number_descending_blocks))
            print('Number of times to repeat a block = {}'.format(self.number_repeating_blocks))
            print('The fully connected layer to repeat - {}'.format(self.repeating_blockd_size))
        
    def restart_network(self):
        self.dataset = utils.MNIST_for_MINE(train=self.train_value)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,  batch_size = self.batch_size, shuffle = True) 
        if self.net_num == 1 :
            self.net = networks.statistical_estimator_DCGAN(input_size = 2, output_size = 1)
        elif self.net_num == 2:
            self.net = networks.statistical_estimator_DCGAN_2(input_size = 1, output_size = 1)
        else:
            
            self.net = networks.statistical_estimator_DCGAN_3(input_size = 1, output_size = 1,
                                                              number_descending_blocks = self.number_descending_blocks, 
                 number_repeating_blocks=self.number_repeating_blocks, repeating_blockd_size = self.repeating_blockd_size)
        print('')
        print('Restarted Network')
        if self.optimizer == 1:
            self.mine_net_optim = optim.SGD(self.net.parameters(), lr = self.lr)
            print('Optimizer: SGD')
        elif self.optimizer == 2: 
            self.mine_net_optim = optim.Adam(self.net.parameters(), lr = self.lr)
            print('Optimizer: Adam')
        else:
            self.mine_net_optim = optim.RMSprop(self.net.parameters(), lr = self.lr)
            print('Optimizer: RMSprop')
        #self.scheduler = optim.lr_scheduler.StepLR(self.mine_net_optim, step_size=10*(len(self.dataset)/self.batch_size), gamma=self.gamma)
        print('Learning rate = {}'.format(self.lr))
        print('Batch size = {}'.format(self.batch_size))
        #print('Gamma of lr decay = {}'.format(self.gamma))
        print('Using Net {}'.format(self.net_num))
        if self.net_num == 3:
            print('Number of Descending Blocks is {}'.format(self.number_descending_blocks))
            print('Number of times to repeat a block = {}'.format(self.number_repeating_blocks))
            print('The fully connected layer to repeat - {}'.format(self.repeating_blockd_size))
            
            
    def mutual_information(self,joint1, joint2, marginal):
        T = self.net(joint1, joint2)
        eT = torch.exp(self.net(joint1, marginal))
        NIM = torch.mean(T) - torch.log(torch.mean(eT)) #The neural information measure by the Donskar-Varadhan representation
        return NIM, T, eT

    def learn_mine(self,batch, ma_rate=0.01):
        # batch is a tuple of (joint1, joint2, marginal (from the dataset of joint 2))
        joint1 = torch.autograd.Variable(batch[0])
        joint2 = torch.autograd.Variable(batch[2])
        marginal = torch.autograd.Variable(batch[4]) #the uneven parts of the dataset are the labels 
        #joint = torch.autograd.Variable(torch.FloatTensor(joint))
        #marginal = torch.autograd.Variable(torch.FloatTensor(marginal))
        
        NIM , T, eT = self.mutual_information(joint1, joint2, marginal)
        
        #Using exponantial moving average to correct bias 
        ma_eT = (1-ma_rate)*eT + (ma_rate)*torch.mean(eT) 
        # unbiasing 
        loss = -(torch.mean(T) - (1/ma_eT.mean()).detach()*torch.mean(eT))
        # use biased estimator
        # loss = - mi_lb
        
        self.mine_net_optim.zero_grad()
        autograd.backward(loss)
        self.mine_net_optim.step()
        #self.scheduler.step()
        self.scheduler2.step(NIM)
        
        return NIM, loss
    
    def train(self, epochs=1):
        # data is x or y
        
        
        result = list()
        bar = pyprind.ProgBar(len(self.dataloader)*epochs, monitor = True)
        nan = None 
        
        for i in range(epochs):
            self.dataset = utils.MNIST_for_MINE(train=self.train_value)
            self.dataloader = torch.utils.data.DataLoader(self.dataset,  batch_size = self.batch_size, shuffle = True)    
            temp_results = []
            for idx, batch in enumerate(self.dataloader):
                mi_lb, loss = self.learn_mine(batch)
                temp_results.append(mi_lb.detach())
                if torch.isnan(temp_results[-1]):
                    print(temp_results[-6:-1])
                    print('Got to NaN in epoch-{0} and batch {1}'.format(i,idx))
                    #plt.plot(result)
                    nan = True
                    break
                bar.update()
            if nan:
                break
            
            result.append(np.mean(temp_results))
                
        return result
    
    def ma(a, window_size=100):
        return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]


        
    

