
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
    def __init__(self):
        
        self.net = networks.statistical_estimator_DCGAN(input_size = 2, output_size = 1)
        #self.input1 = input1
        print('start')
        #self.input2 = input2
        self.mine_net_optim = optim.Adam(self.net.parameters(), lr = 3e-3)
        
        
    
        self.dataset = utils.MNIST_for_MINE(train=True)
        
        self.dataloader = torch.utils.data.DataLoader(self.dataset,  batch_size = 1000, shuffle = True)    
        
            
            
    
    def mutual_information(self,joint1, joint2, marginal):
        t = self.net(joint1, joint2)
        et = torch.exp(self.net(joint1, marginal))
        mi_lb = torch.mean(t) - torch.log(torch.mean(et))
        return mi_lb, t, et

    def learn_mine(self,batch, ma_et, ma_rate=0.01):
        # batch is a tuple of (joint, marginal)
        joint1 , joint2, marginal = batch[0], batch[2], batch[4]
        #joint = torch.autograd.Variable(torch.FloatTensor(joint))
        #marginal = torch.autograd.Variable(torch.FloatTensor(marginal))
        mi_lb , t, et = self.mutual_information(joint1, joint2, marginal)
        ma_et = (1-ma_rate)*ma_et + ma_rate*torch.mean(et)
        
        # unbiasing use moving averpyprindage
        loss = -(torch.mean(t) - (1/ma_et.mean()).detach()*torch.mean(et))
        # use biased estimator
        # loss = - mi_lb
        
        self.mine_net_optim.zero_grad()
        autograd.backward(loss)
        self.mine_net_optim.step()
        return mi_lb, ma_et
    
    def train(self, batch_size=100, epochs=1, log_freq=int(1e+3)):
        # data is x or y
        result = list()
        ma_et = 1
        bar = pyprind.ProgBar(len(self.dataloader)*epochs, monitor = True)
        for i in range(epochs):
            temp_results = []
            for idx, batch in enumerate(self.dataloader):
                mi_lb, ma_et = self.learn_mine(batch, ma_et)
                temp_results.append(mi_lb.detach().numpy())
                if not mi_lb.detach().numpy():
                    print(mi_lb.detach())
                    plt.plot(result)
                    break
                if (i+1)%(log_freq)==0:
                    print(result[-1])
                bar.update()
            result.append(np.mean(temp_results))
                
        return result
    
    def ma(a, window_size=100):
        return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]


        
    

