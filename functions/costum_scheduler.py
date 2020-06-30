#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:42:15 2019
contains costum learning rate scheduler. At the moment I have implemented
a cyclic learning rate schedule. 
@author: danny
"""
from torch.optim import lr_scheduler
import numpy as np

# cyclic scheduler which varies the learning rate between a min and max over a 
# certain number of epochs according to a cosine function. Operates between 1 
# and 3 (so cos cycles from -1 to -1 ) normalises this between 0 and 1 and then 
# presses between min and max lr 
def cyclic_scheduler(max_lr, min_lr, stepsize, optimiser):
    lr_lambda = lambda iteration: (max_lr - min_lr) * (.5 * (np.cos(np.pi * (1 + (3 - 1) / stepsize * iteration)) + 1)) + min_lr
    cyclic_scheduler = lr_scheduler.LambdaLR(optimiser, lr_lambda, 
                                             last_epoch = -1)
  
    return(cyclic_scheduler)
