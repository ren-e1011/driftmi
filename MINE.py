
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
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torchvision

import networks 
import utils



class MINE():
    def __init__(self, train = True, batch = 1000, lr = 3e-3):
        
        self.net = networks.statistical_estimator_DCGAN(input_size = 2, output_size = 1)
        #self.input1 = input1
        #self.input2 = input2
        self.mine_net_optim = optim.SGD(self.net.parameters(), lr = lr)
        
        
    
        self.dataset = utils.MNIST_for_MINE(train=train)
        
        self.dataloader = torch.utils.data.DataLoader(self.dataset,  batch_size = batch, shuffle = True)    
        
            
            
    
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
        return NIM, loss
    
    def train(self, batch_size=100, epochs=1, log_freq=int(1e+3)):
        # data is x or y
        result = list()
        bar = pyprind.ProgBar(len(self.dataloader)*epochs, monitor = True)
        nan = None 
        for i in range(epochs):
            temp_results = []
            for idx, batch in enumerate(self.dataloader):
                mi_lb, loss = self.learn_mine(batch)
                temp_results.append(mi_lb.detach())
                if torch.isnan(temp_results[-1]):
                    print(i)
                    plt.plot(result)
                    nan = True
                    break
                if (i+1)%(log_freq)==0:
                    print(result[-1])
                bar.update()
            if nan:
                break
            result.append(np.mean(temp_results))
                
        return result
    
    def ma(a, window_size=100):
        return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]


        
    

