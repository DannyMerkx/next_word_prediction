#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 14:49:30 2018

@author: danny
"""
from mini_batcher import char_batcher, token_batcher
from grad_tracker import gradient_clipping
from torch.autograd import Variable

import string
import torch
import os
import time
import pickle
import numpy as np 

# trainer for the flickr database. 
class nwp_trainer():
    def __init__(self, encoder):
        # default datatype, change to cuda by calling set_cuda
        self.dtype = torch.FloatTensor
        # set the transformer. Set an empty scheduler to keep this optional.
        self.encoder = encoder
        self.scheduler = False
        # set gradient clipping to false by default
        self.grad_clipping = False
        # set the attention loss to empty by default
        self.att_loss = False
        # keep track of an iteration for lr scheduling
        self.iteration = 0
        # keep track of the number of training epochs
        self.epoch = 1
        
    # the possible minibatcher for all different types of data for the databases (tokens and chars)
    def char_batcher(self, sents, batch_size, max_len, shuffle):
        return char_batcher(sents, batch_size, max_len, shuffle)
    def token_batcher(self, sents, batch_size, max_len, shuffle):
        return token_batcher(sents, batch_size, self.dict_loc, max_len, shuffle)
################## functions to set class values and attributes ###############
    # functions to set which minibatcher to use. Needs to be called as no default is set.
    def set_char_batcher(self):
        self.batcher = self.char_batcher
    def set_token_batcher(self):
        self.batcher = self.token_batcher
    # function to set the learning rate scheduler and type (for deciding when to update the schedule etc.)
    def set_lr_scheduler(self, scheduler, s_type):
        self.lr_scheduler = scheduler  
        self.scheduler = s_type
    # function to set the loss for training. Loss is not necessary e.g. when you 
    # only want to test a pretrained model.
    def set_loss(self, loss):
        self.loss = loss
    # set an optimizer. Optional like the loss in case of using just pretrained models.
    def set_optimizer(self, optim):
        self.optimizer = optim
    # set a dictionary. for models trained on tokens
    def set_dict_loc(self, loc):
        self.dict_loc = loc
    # set data type and the network to cuda
    def set_cuda(self):
        self.dtype = torch.cuda.FloatTensor
        self.encoder.cuda()
    # manually set the epoch to some number e.g. if continuing training from a 
    # pretrained model
    def set_epoch(self, epoch):
        self.epoch = epoch
    def update_epoch(self):
        self.epoch += 1
    # functions to set new embedders
    def set_encoder(self, emb):
        self.encoder = emb
    # functions to load a pretrained embedder
    def load_encoder(self, loc):
        enc_state = torch.load(loc)
        self.encoder.load_state_dict(enc_state)
    # optionally load glove embeddings for token based embedders with load_embeddings
    # function implemented.
    def load_glove_embeddings(self, glove_loc):
        self.encoder.load_embeddings(self.dict_loc, glove_loc)

################## functions to perform training and testing ##################
        
# during training the model expects aligned input from both languages. The model
# outputs predictions (next word probabilities for each position) and targets for the loss function
# (the decoder input shifted to the left)
    def train_epoch(self, sents, batch_size, save_states, save_loc):
        print('training epoch: ' + str(self.epoch))
        # keep track of runtime
        self.start_time = time.time()
        self.encoder.train()
        # for keeping track of the average loss over all batches
        self.train_loss = 0
        num_batches = 0
        for batch in self.batcher(sents, batch_size, 
                                  self.encoder.max_len, shuffle = True):
            # retrieve a minibatch from the batcher
            enc_input, lengths = batch
            num_batches +=1
            # embed the images and audio using the network
            preds, targs = self.encode(enc_input, lengths)
            # calculate the loss            
            loss = self.loss(preds.view(-1, preds.size(-1)), targs.view(-1))
            # reset the gradients of the optimiser
            self.optimizer.zero_grad()
            # calculate the gradients and perform the backprop step
            loss.backward()
            # clip the gradients if required
            if self.grad_clipping:
                torch.nn.utils.clip_grad_norm(self.encoder.parameters(), self.tf_clipper.clip)
            # update weights
            self.optimizer.step()
            # add loss to average
            self.train_loss += loss.data
            # print loss every n batches
            if int(num_batches * batch_size) in save_states:
                print(' '.join(['loss after', str(num_batches), 'sentences:', str(self.train_loss.cpu().data.numpy()/num_batches)]))
                self.save_params(save_loc, int(num_batches * batch_size))
            # if there is a cyclic lr scheduler, take a step in the scheduler
            if self.scheduler == 'cyclic':
                self.lr_scheduler.step()
                self.iteration +=1     
        # print the average loss for the current epoch
        self.train_loss = self.train_loss.cpu().data.numpy()/num_batches
# during testing the model takes encoder input and translates the sentence into 
# the target language without looking at the decoder input. decoder input is therefore
# optional. If it is passed, this is just used to create the gold label targets for calculating 
# the test loss.
    def test_epoch(self, sents, batch_size, beam_width = 1):
        # set to evaluation mode (disable dropout etc.)
        self.encoder.eval()
        # for keeping track of the average loss
        self.test_loss = 0
        test_batches = 0
        for batch in self.batcher(sents, batch_size,
                                  self.encoder.max_len, shuffle = False):
            # retrieve a minibatch from the batcher
            enc_input, lengths = batch
            test_batches += 1      
            # translate the sentences and return the candidate translation, prediction probabilities and targets
            preds, targs = self.encode(enc_input, lengths)
            # calculate the cross entropy loss
            loss = self.loss(preds.view(-1, preds.size(-1)), targs.view(-1))
            # add loss to average
            self.test_loss += loss.data 
        self.test_loss = self.test_loss.cpu().data.numpy()/test_batches
        # take a step for a plateau lr scheduler                
        if self.scheduler == 'plateau':
            self.lr_scheduler.step(self.test_loss)
    # embed a batch of images and captions
    def encode(self, enc, l):  
	# sort the minibatch by length
        enc = enc[np.argsort(- np.array(l))]
        l = np.array(l)[np.argsort(- np.array(l))] 
        # convert data to pytorch variables
        enc = self.dtype(enc)    
        # call the transformer's forward function
        preds, targs = self.encoder(enc, l)
        return preds, targs

######################## evaluation functions #################################
    # report on the time this epoch took and the train and test loss
    def report(self, max_epochs):
        # report on the time and train and val loss for the epoch
        print("Epoch {} of {} took {:.3f}s".format(
                self.epoch, max_epochs, time.time() - self.start_time))
        self.print_train_loss()
        self.print_validation_loss()
    # print the loss values
    def print_train_loss(self):  
        print("training loss:\t\t{:.6f}".format(self.train_loss))
    def print_test_loss(self):        
        print("test loss:\t\t{:.6f}".format(self.test_loss))
    def print_validation_loss(self):
        print("validation loss:\t\t{:.6f}".format(self.test_loss))
    # function to save parameters in a results folder
    def save_params(self, loc, num_batches):
        torch.save(self.encoder.state_dict(), os.path.join(loc, '_'.join(['nwp_model', str(self.epoch), str(num_batches)])))
############ functions to deal with the trainer's gradient clipper ############
    # create a gradient tracker/clipper
    def set_gradient_clipping(self, tf_value):
        self.grad_clipping = True
        self.tf_clipper = gradient_clipping(tf_value)
        self.tf_clipper.register_hook(self.encoder)

    # save the gradients collected so far 
    def save_gradients(self, loc):
        self.tf_clipper.save_grads(loc, 'nwp_grads')
    # reset the grads for a new epoch
    def reset_grads(self):
        self.tf_clipper.reset_gradients()
    # update the clip value of the gradient clipper based on the previous epoch. Don't call after resetting
    # the grads to 0
    def update_clip(self):
        self.tf_clipper.update_clip_value()
