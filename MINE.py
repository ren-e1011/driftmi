
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


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd


class MINE():
    def __init__(self, input1, input2):
        self.what = 0
        
    class statistical_estimator_big(nn.Module):
        def __init__(self, input_size = 1, classifier = False):
            super().__init__()
            # Define the networks steps:
            self.conv1 = nn.Conv2d(1, 32, 3, stride = 1, padding = 1)
            self.batchNorm1 = nn.BatchNorm2d(32, momentum=1)
            self.relu = nn.ReLU(inplace = True)
            self.pool1 = nn.MaxPool2d(2,2)
            self.conv2 = nn.Conv2d(32, 64, 3, stride = 1, padding = 1)
            self.batchNorm2 = nn.BatchNorm2d(64, momentum=1)
            self.pool2 = nn.MaxPool2d(2,2)
            self.conv3 = nn.Conv2d(64, 64, 3, stride = 1, padding = 1)
            self.batchNorm3 = nn.BatchNorm2d(64, momentum=1)
            self.pool3 = nn.MaxPool2d(2,2)
            self.conv4 = nn.Conv2d(64, 64, 3, stride = 1, padding = 1)
            self.batchNorm4 = nn.BatchNorm2d(64, momentum=1)
            self.pool4 = nn.MaxPool2d(2,2)
            if classifier:
                self.fc = nn.Linear(64,10)
            else:
                self.fc = nn.Linear(64,1)
            #self.softmax = nn.Softmax(dim = 0)
        
        def forward(self, x):
            #x = torch.from_numpy(x)
            x = self.pool1(self.relu(self.batchNorm1(self.conv1(x))))
            x = self.pool2(self.relu(self.batchNorm2(self.conv2(x))))
            x = self.pool3(self.relu(self.batchNorm3(self.conv3(x))))
            x = self.pool4(self.relu(self.batchNorm4(self.conv4(x))))
            x = x.view(x.size(0), 64)
            x = self.fc(x)
            #x = self.softmax(x)
            
            return x
        
    class statistical_estimator_small(nn.Module):
        def __init__(self, input_size = 1, classifier = False):
            super().__init__()
            # Define the networks steps:
            self.conv1 = nn.Conv2d(1, 32, 3, stride = 1, padding = 1)
            self.batchNorm1 = nn.BatchNorm2d(32, momentum=1)
            self.relu = nn.ReLU(inplace = True)
            self.pool1 = nn.MaxPool2d(2,2)
            self.conv2 = nn.Conv2d(32, 64, 3, stride = 1, padding = 1)
            self.batchNorm2 = nn.BatchNorm2d(64, momentum=1)
            self.pool2 = nn.MaxPool2d(2,2)
            if classifier:
                self.fc = nn.Linear(64*6,10)
            else:
                self.fc = nn.Linear(64*6,1)
            #self.softmax = nn.Softmax(dim = 0)
        
        def forward(self, x):
            #x = torch.from_numpy(x)
            x = self.pool1(self.relu(self.batchNorm1(self.conv1(x))))
            x = self.pool2(self.relu(self.batchNorm2(self.conv2(x))))
            x = self.pool3(self.relu(self.batchNorm3(self.conv3(x))))
            x = self.pool4(self.relu(self.batchNorm4(self.conv4(x))))
            x = x.view(x.size(0), 64)
            x = self.fc(x)
            #x = self.softmax(x)
            
            return x
        
    def mutual_information(self,joint, marginal, mine_net):
        t = mine_net(joint)
        et = torch.exp(mine_net(marginal))
        mi_lb = torch.mean(t) - torch.log(torch.mean(et))
        return mi_lb, t, et

    def learn_mine(self,batch, mine_net, mine_net_optim,  ma_et, ma_rate=0.01):
        # batch is a tuple of (joint, marginal)
        joint , marginal = batch
        joint = torch.autograd.Variable(torch.FloatTensor(joint)).cuda()
        marginal = torch.autograd.Variable(torch.FloatTensor(marginal)).cuda()
        mi_lb , t, et = self.mutual_information(joint, marginal, mine_net)
        ma_et = (1-ma_rate)*ma_et + ma_rate*torch.mean(et)
        
        # unbiasing use moving average
        loss = -(torch.mean(t) - (1/ma_et.mean()).detach()*torch.mean(et))
        # use biased estimator
    #     loss = - mi_lb
        
        mine_net_optim.zero_grad()
        autograd.backward(loss)
        mine_net_optim.step()
        return mi_lb, ma_et
        
    
    def augment_photoes(data):
        '''
        Parameters
        ----------
        image : ndarray
            Input image data. Will be converted to float.
        mode : str
            One of the following strings, selecting the type of noise to add:
        
            'gauss'     Gaussian-distributed additive noise.
            'poisson'   Poisson-distributed noise generated from the data.
            's&p'       Replaces random pixels with 0 or 1.
            'speckle'   Multiplicative noise using out = image + n*image,where
                        n is uniform noise with specified mean & variance.
        
        '''
        
        import numpy as np
        import os
        import cv2
        def noisy(noise_typ,image):
           if noise_typ == "gauss":
              row,col,ch= image.shape
              mean = 0
              var = 0.1
              sigma = var**0.5
              gauss = np.random.normal(mean,sigma,(row,col,ch))
              gauss = gauss.reshape(row,col,ch)
              noisy = image + gauss
              return noisy
           elif noise_typ == "s&p":
              row,col,ch = image.shape
              s_vs_p = 0.5
              amount = 0.004
              out = np.copy(image)
              # Salt mode
              num_salt = np.ceil(amount * image.size * s_vs_p)
              coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
              out[coords] = 1
        
              # Pepper mode
              num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
              coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
              out[coords] = 0
              return out
          elif noise_typ == "poisson":
              vals = len(np.unique(image))
              vals = 2 ** np.ceil(np.log2(vals))
              noisy = np.random.poisson(image * vals) / float(vals)
              return noisy
          elif noise_typ =="speckle":
              row,col,ch = image.shape
              gauss = np.random.randn(row,col,ch)
              gauss = gauss.reshape(row,col,ch)        
              noisy = image + image * gauss
              return noisy


