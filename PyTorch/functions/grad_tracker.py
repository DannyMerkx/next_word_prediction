# -*- coding: utf-8 -*-
"""
Created on Fri May  4 09:42:05 2018
gradient clipping class, can be added to the trainer to allow for gradient clipping
@author: danny
"""
import numpy as np

# create a gradient clipping object, so you can set a backward hook on a model,
# keep track of the gradients for each epoch or the total session and keep a clipping
# value to use for gradient clipping
class gradient_clipping():
    def __init__(self, clip_value):
        # keep track of the gradients per epoch
        self.epoch_grads = []
        # list to append the gradients to after each epoch
        self.total_grads = []
        # register an initial clipping value
        self.clip = clip_value
    # this appends the gradient norm at each backward call. x is a dummy because 
    # the backward hook passes the model's self.
    def track_grads(self, x, grad_input, grad_output):
        self.epoch_grads.append(grad_input[0].norm().cpu().data.numpy())
    # register a backward hook to the encoder            
    def register_hook(self, encoder):
        encoder.register_backward_hook(self.track_grads)
    # return the gradient norm mean over the epoch
    def gradient_mean(self):
        return np.mean(self.epoch_grads)
    def gradient_std(self):
        return np.std(self.epoch_grads)
    # reset the epoch gradients
    def reset_gradients(self):
        self.total_grads.append(self.epoch_grads)        
        self.epoch_grads = []
    # save the gradients of the entire training loop
    def save_grads(self, loc, name):
        np.save(loc + name, self.total_grads)
    # update the clipping value based on the running gradient mean and standard deviation for the previous epoch
    def update_clip_value(self):       
        self.clip = self.gradient_mean() + self.gradient_std()
    def update_clip_value_total(self):
        # add the running epoch to the total grad list
        grads = [y for x in self.total_grads.append(self.epoch_grads) for y in x]
        self.clip = np.mean(grads)

    
