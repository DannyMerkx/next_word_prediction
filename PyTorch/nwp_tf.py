#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 13:01:05 2018
This script creates and trains a next word predictor using the Transformer encoder. For next word prediction
we use only the encoder part of the transformer but we do use a decoder mask (to prevent peaking at future 
timesteps making NWP trivial)

@author: danny
"""
from __future__ import print_function
from torch.optim import lr_scheduler

import argparse
import torch
import numpy as np
import sys
import os
import pickle
import csv
sys.path.append('./functions')

from encoders import nwp_transformer, nwp_transformer_2lin
from nwp_trainer import nwp_trainer
from costum_scheduler import cyclic_scheduler

parser = argparse.ArgumentParser(description = 'Create and run an articulatory feature classification DNN')

# args concerning file location
parser.add_argument('-data_loc', type = str, 
                    default = '/data/databases/next_word_prediction/',
                    help = 'location of the training sentences')
parser.add_argument('-results_loc', type = str, 
                    default = '/data/next_word_prediction/PyTorch/tf_results/',
                    help = 'location to save the trained network parameters')
parser.add_argument('-dict_loc', type = str, 
                    default = '/data/next_word_prediction/PyTorch/nwp_indices',
                    help = 'location of dictionary mapping the vocabulary to embedding indices')
# args concerning training settings
parser.add_argument('-batch_size', type = int, default = 10, 
                    help = 'batch size, default: 10')
parser.add_argument('-lr', type = float, default = 0.005,
                    help = 'learning rate, default:0.005')
parser.add_argument('-n_epochs', type = int, default = 8, 
                    help = 'number of training epochs, default: 8')
parser.add_argument('-cuda', type = bool, default = True, 
                    help = 'use cuda (gpu), default: True')
parser.add_argument('-save_states', type = list, 
                    default = [1000, 3000, 10000, 30000, 100000, 300000, 
                               1000000, 3000000, 5855670], 
                    help = '#sentences after which model parameters are saved')

parser.add_argument('-gradient_clipping', type = bool, default = False,
                    help ='use gradient clipping, default: False')
parser.add_argument('-seed', type = list, default = [745546129, 1936929273], 
                    help = 'optional seed for the random components')

args = parser.parse_args()

# check if cuda is available and if user wants to run on gpu
cuda = args.cuda and torch.cuda.is_available()
if cuda:
    print('using gpu')
else:
    print('using cpu')

# check is there is a given random seed (list!). If not create one but print it 
# so it can be used to replicate this run. 
if args.seed:
    np.random.seed(args.seed[0])
    torch.manual_seed(args.seed[1])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    seed = np.random.randint(0, 2**32, 2)
    print('random seeds (numpy, torch): ' + str(seed))
    np.random.seed(seed[0])
    torch.manual_seed(seed[1])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)
    
# get the size of the dictionary and add 1 for the zero or padding embedding
dict_size = len(load_obj(args.dict_loc)) + 1 
# config settings for the transformer
config = {'embed': {'n_embeddings': dict_size,'embedding_dim': 400, 
                    'sparse': False, 'padding_idx':0
                    }, 
          'tf':{'in_size':400, 'fc_size': 1024,'n_layers': 1,'h': 8, 
                'max_len': 52
                },  
          'cuda': cuda
          }

def load(folder, file_name):
    open_file = open(os.path.join(folder, file_name))
    line = [x for x in open_file]
    open_file.close()  
    return line  
    
def load_index(folder, file_name):
    open_file = open(os.path.join(folder, file_name))
    reader = csv.reader(open_file, delimiter = ',')
    line = [[int(y) for y in x] for x in reader]
    open_file.close()
    return line

train = load(args.data_loc, 'train_nwp.txt')
print(f'learning rate: {args.lr}')

####################### Neural network setup ##################################
# create the network and initialise the parameters to be xavier uniform 
# distributed
nwp_model = nwp_transformer(config)

for p in nwp_model.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)
    elif p.dim() <=1:
        torch.nn.init.normal_(p)

model_parameters = filter(lambda p: p.requires_grad, nwp_model.parameters())
print(f'#model parameters: {sum([np.prod(p.size()) for p in model_parameters])}')

# optimiser for the network
optimizer = torch.optim.SGD(nwp_model.parameters(), lr = args.lr, momentum = .9)

#plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.2, patience = 0, 
#                                                   threshold = 0.0001, min_lr = 1e-5, cooldown = 0)

# set the step size for the learning rate scheduler to be 1/3 of the data.
step_size = int(len(train)/(3 * args.batch_size))
step_scheduler = lr_scheduler.StepLR(optimizer, step_size, gamma=.5, 
                                     last_epoch=-1)

# create a trainer setting the loss function, optimizer, minibatcher and lr_scheduler
trainer = nwp_trainer(nwp_model)
trainer.set_dict_loc(args.dict_loc)
trainer.set_loss(torch.nn.CrossEntropyLoss(ignore_index= 0))
trainer.set_optimizer(optimizer)
trainer.set_token_batcher()
trainer.set_lr_scheduler(step_scheduler, 'cyclic')

#optionally use cuda and gradient clipping
if cuda:
    trainer.set_cuda()

# gradient clipping can help stabilise training in the first epoch.
if args.gradient_clipping:
    trainer.set_gradient_clipping(0.25)

############################# training/test loop ##############################
# run the training loop for the indicated amount of epochs 
while trainer.epoch <= args.n_epochs:
    # Train on the train set    
    trainer.train_epoch(train, args.batch_size, args.save_states, 
                        args.results_loc)

    if args.gradient_clipping:
        trainer.reset_grads()
    # increase epoch#
    trainer.update_epoch()
    # reset the model for the next epoch
    for p in nwp_model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
        elif p.dim() <=1:
            torch.nn.init.normal_(p)
    optimizer = torch.optim.SGD(nwp_model.parameters(), lr = args.lr, momentum = .9)
    step_scheduler = lr_scheduler.StepLR(optimizer, step_size, gamma=0.5, 
                                         last_epoch = -1)

    trainer.set_optimizer(optimizer)
    trainer.set_lr_scheduler(step_scheduler, 'cyclic')

# save the gradients for each epoch, can be useful to select an initial 
# clipping value.
if args.gradient_clipping:
    trainer.save_gradients(args.results_loc)