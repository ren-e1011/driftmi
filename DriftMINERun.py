import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import MINE
import sys
import os
import logging
from random import shuffle, choice

import FullDriftDataset
# README.md
# An example of a run:​

# bsub -q gpu-short -app nvidia-gpu -env LSB_CONTAINER_IMAGE=nvcr.io/nvidia/pytorch:19.07-py3 -gpu num=4:j_exclusive=no -R "select[mem>8000] rusage[mem=8000]" -o out.%J -e err.%J python3 \
# Adam, {lr}, batch_size, 
# mine_run.py 2 0.0003 500 200 1 2 combined 3 2 2 512 6 same
# ---

# sys.argv gets 4 values:
# [1] - the optimizer - one of three (1) - SGD,(2) - Adam, (3) - RMSprop
# [2] - lr
# [3] - batch_size
# [4] - epochs
# [5] - train true/false 1/0
# [6] - Net num - 1-3 #set equal 3 Ignore
# [7] - traject/MNIST/combined - What form of mine to compute # combined Ignore
# [8] - number_descending_blocks - i.e. how many steps should the fc networks take,
#                               starting at 2048 nodes and every step divide the size 
#                                by 2. max number is 7                                
# [9] - number_repeating_blocks - the number of times to repeat a particular layer
# [10] - repeating_blockd_size - the size of nodes of the layer to repeat
# [11] - traject_max_depths
# [12] - traject_num_layers
# [13] - same/different minist/trajectory - to use the same image that the 
#        trajectory ran on as the joint distribution. # Ignore? 

BASE_PATH = '/Volumes/Experiment Data/Sasha and Or/'
part_set = set([_dir for _dir in os.listdir(BASE_PATH) if not _dir.startswith(".")])
exclude_set = set(['DL','OL','SM'])
PART_LIST = list(part_set - exclude_set)
PART_LIST = [p for p in PART_LIST if 'try' not in p.lower()]

# 300 observations per participant
NSTIMULI = 100
NOBS = 300

GAZE_COLS = ['norm_pos_x','norm_pos_y']
HEAD_COLS = ['head rot x','head rot y','head rot z','head_dir_x','head_dir_y','head_dir_z','head_right_x','head_right_y','head_right_z','head_up_x','head_up_y','head_up_z']
NCHANNELS = len(GAZE_COLS)+len(HEAD_COLS)

# selection = {i:np.array(range(0,NSTIMULI)) for i in range(len(PART_LIST))}

# for testing
PART_LIST = PART_LIST[:3]
print('Testing on participants,',PART_LIST)
selection = {participant:np.array(range(0,NSTIMULI)) for participant in PART_LIST}
for k in selection.keys():
    # shuffle each individually 
    # inplace function. returns None
    shuffle(selection[k])
# train set
# 240 random observations of each participant
cut1 = int(NSTIMULI * .80)
train_selection = {k:selection[k][:cut1] for k in selection.keys()}
# validation set
# 30 random observations of each participant
cut2 = int(NSTIMULI * .90)
val_selection = {k:selection[k][cut1:cut2] for k in selection.keys()}
# test set
# 30 random observations of each participant
test_selection = {k:selection[k][cut2:] for k in selection.keys()}

# TODO mod input params 
gamma = 0
# optimizer = int(sys.argv[1])
optimizer = 2 # Adam
# lr = float(sys.argv[2])
lr = 0.0003 
# batch_size = int(sys.argv[3])
batch_size = 16
# epochs = int(sys.argv[4])
epochs = 2 # for testing
# if int(sys.argv[5]) == 1:
#     train = True
# else:
#     train = False
# TODO mod test (train=False) for net
train = True
# hard code. only combined makes sense
net_num = 1
# traject = str(sys.argv[7])
# number_descending_blocks = int(sys.argv[8])
# number_repeating_blocks = int(sys.argv[9])
# repeating_blockd_size = int(sys.argv[10])
# traject_max_depth = int(sys.argv[11])
# traject_max_depth = 128
# traject_num_layers = int(sys.argv[12])
# traject_num_layers = 3
# first_depth = 32
# dataset_status = str(sys.argv[13])
mine = MINE.MINE(train = True, 
                 batch = batch_size, 
                 lr = lr,
                 gamma = gamma, 
                 optimizer = optimizer,
                #  traject_max_depth = traject_max_depth, 
                #  traject_num_layers = traject_num_layers, 
                 traject_stride = [3,1],
                 traject_kernel = 5, 
                 traject_padding = 0,
                 traject_pooling = [1,2], 
                #  number_descending_blocks = number_descending_blocks, 
                #  number_repeating_blocks=number_repeating_blocks,  
                #  repeating_blockd_size = repeating_blockd_size,
                #  dataset_status = dataset_status,
                 traject_input_dim = [NCHANNELS,NOBS])

print(mine.net)

safety = 0
# print('Epochs: {}'.format(int(sys.argv[4])))

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# ? 
torch.backends.cudnn.benchmark = True

dataset = FullDriftDataset.FullDataset(ix_dict=train_selection)
# num_workers maybe comment out
params = {'batch_size': batch_size,
          'shuffle': False}
        #   'num_workers': 6}
traj_generator = torch.utils.data.DataLoader(dataset, **params)

val_dataset = FullDriftDataset.FullDataset(ix_dict=val_selection)
params = {
    'batch_size':batch_size,
    'shuffle':False
}
val_generator = torch.utils.data.DataLoader(val_dataset, **params)


# opening MINE.train() which calls MINE.epoch() which calls MINE.learn_mine()
results = []
losses = []
for epoch in range(epochs):
# Train
    epoch_results = []
    epoch_losses = []
    for i, sample in enumerate(traj_generator):
        trajectory,joint,marginal = sample
        traj_inp = trajectory.permute(0,2,1).float()
        joint_inp = joint.permute(0,3,1,2).float()
        marg_inp = marginal.permute(0,3,1,2).float()

        traj_inp, joint_inp, marg_inp = traj_inp.to(device), joint_inp.to(device), marg_inp.to(device)

        # where is loss recorded, managed
        NIM, loss = mine.learn_mine((traj_inp,joint_inp,marg_inp))
        if torch.isnan(NIM):
            ix = batch_size * i
            # which samples 
            print('NaN epoch {0}:: samples {1}'.format(epoch, dataset.ix_list[ix:ix+batch_size]))
            continue
        else:
            epoch_results.append(NIM.detach())
            epoch_losses.append(loss.detach())

    results.append(np.mean(epoch_results))
    losses.append(np.mean(epoch_losses))
    # if nan across all iterations - in what cases?
    if len(results) == 0:
        if safety > 5: 
            print('Nans for 5 epochs. Halting Training')
            break
        safety += 1
        mine.restart_network()

# Validate 
    for val_i, val_sample in enumerate(val_generator):
        trajectory,joint,marginal = sample
        traj_inp = trajectory.permute(0,2,1).float()
        joint_inp = joint.permute(0,3,1,2).float()
        marg_inp = marginal.permute(0,3,1,2).float()
        
        traj_inp, joint_inp, marg_inp = traj_inp.to(device), joint_inp.to(device), marg_inp.to(device)

         # where is loss recorded, managed
        NIM, loss = mine.learn_mine((traj_inp,joint_inp,marg_inp))
        if torch.isnan(NIM):
            ix = batch_size * i
            # which samples 
            print('NaN epoch {0}:: samples {1}'.format(epoch, dataset.ix_list[ix:ix+batch_size]))
            continue
        else:
            epoch_results.append(NIM.detach())
            epoch_losses.append(loss.detach())

    results.append(np.mean(epoch_results))
    losses.append(np.mean(epoch_losses))

        # where is loss recorded, managed
        # NIM, loss = mine.learn_mine((traj_inp,joint_inp,marg_inp))
        # if torch.isnan(NIM):
        #     ix = batch_size * i
        #     # which samples 
        #     print('NaN epoch {0}:: samples {1}'.format(epoch, dataset.ix_list[ix:ix+batch_size]))
        #     continue
        # else:
        #     epoch_results.append(NIM.detach())
        #     epoch_losses.append(loss.detach())

# mine.train(epochs = epochs)
# while len(mine.results) == 0:
#     mine.restart_network()
#     check_trial = mine.train(epochs = epochs)
#     safty+=1
#     if safty>5:
#         break
# safty = 0
# results = pd.DataFrame(mine.results)
results = pd.DataFrame(results)
# nm = 'results_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}'.format(sys.argv[1],int(float(sys.argv[2])*100000),sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8],sys.argv[9],sys.argv[10],dataset_status)
name = 'results_test'
results.to_pickle(name)
#results.to_csv('results_{0}_{1}_{2}_{3}_{4}.csv'.format(sys.argv[1],int(float(sys.argv[2])*10000),sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8]))
torch.save(mine.net.state_dict, 'net_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}'.format(sys.argv[1],int(float(sys.argv[2])*100000),sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8],sys.argv[9],sys.argv[10],dataset_status))

# test

## random shuffling should get nothing