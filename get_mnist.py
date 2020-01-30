import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import os
import numpy as np

import pyprind
os.chdir('/home/michael/Documents/Scxript/')
from networks import statistical_estimator_big, statistical_estimator_small


mnist_data_train = torchvision.datasets.MNIST('home/michael/Documents/MNIST data', train = True, transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,),(0.3081,))]),download = True)

mnist_data_test = torchvision.datasets.MNIST('home/michael/Documents/MNIST data', train = False, transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,),(0.3081,))]), download = True)
    
#train_loader = torch.utils.data.DataLoader(mnist_data_train, batch_size = 64, shuffle = True)
#test_loader = torch.utils.data.DataLoader(mnist_data_test, batch_size = 1000, shuffle = True)

idx = np.random.randint(0,len(mnist_data_test),100)
mini_set = mnist_data_test
mini_set.targets = mini_set.targets[idx]
mini_set.data = mini_set.data[idx]