#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 18:31:24 2018

This script contains functions that serve as the 'forward' function of the Transformer. This 
brings together the encoder and decoder parts, differentiate between training and test time
situations and provides as 'forward' function for situations where you just use the Transformer
as an encoder. Furthermore this script contains functions to load pre-trained embeddings, create 
the encoder and decoder masks, create the positional embeddings and perform beam search on predicted
sequences. 

@author: danny
"""
from load_embeddings import load_word_embeddings

import torch
import torch.nn as nn
import numpy as np
        
# super class with some functions that might be usefull for multiple transformer based architectures        
class transformer(nn.Module):
    def __init__(self):
        super(transformer, self).__init__()
        pass    
    # optionally load pretrained word embeddings. Takes the dictionary of words occuring in the training data
    # add the location of the embeddings.
    def load_embeddings(self, dict_loc, embedding_loc):
        load_word_embeddings(dict_loc, embedding_loc, self.embed.weight.data)
    
    # function to create the positional embeddings
    def pos_embedding(self, sent_len, d_model):
        pos_emb = torch.zeros(int(sent_len), d_model)
        for x in range(0, sent_len):
            for y in range (0, d_model, 2):
                pos_emb[x, y] = torch.sin(torch.Tensor([x / (10000 ** ((2 * y) / d_model))]))
                pos_emb[x, y + 1] = torch.cos(torch.Tensor([x / (10000 ** ((2 * (y + 1)) / d_model))]))
        if self.is_cuda == True:
            pos_emb = pos_emb.cuda()
        return pos_emb
    
    # create the encoder mask, which masks the padding indices 
    def create_enc_mask(self, input):
        return (input != 0).unsqueeze(1)
    
    # create the decoder mask, which masks the padding and for each timestep
    # all future timesteps
    def create_dec_mask(self, input):
        seq_len = input.size(1)
        # create a mask which masks the padding indices
        mask = (input != 0).unsqueeze(1)
        # create a mask which masks for each time-step the futuru time-steps
        triu = (np.triu(np.ones([1, seq_len, seq_len]), k = 1) == 0).astype('uint8')
        # combine the two masks
        if self.is_cuda == True:
            dtype = torch.cuda.ByteTensor
        else:
            dtype = torch.ByteTensor
        return dtype(triu) & mask
            
    # Function used for training a transformer encoder-decoder, receives both the original 
    # sentence and its translation. Use in the architectures' forward function
    def encoder_decoder_train(self, enc_input, dec_input):
        # create the targets for the loss function (decoder input, shifted to the left, padded with a zero)
        targs = torch.nn.functional.pad(dec_input[:, 1:], [0, 1]).long()

        # create the encoder mask which is 0 where the input is padded along the time dimension
        e_mask = self.create_enc_mask(enc_input)

        # retrieve embeddings for the sentence and scale the embeddings importance relative to the positional embeddings
        e_emb = self.embed(enc_input.long()) * np.sqrt(self.embed.embedding_dim)

        # apply the (stacked) encoder transformer
        encoded = self.TF_enc(e_emb + self.pos_emb[:enc_input.size(1), :], mask = e_mask)  

        # create the decoder mask for padding, which also prevents the decoder from looking into the future.
        d_mask = self.create_dec_mask(dec_input)

         # retrieve embeddings for the sentence and scale the embeddings importance relative to the positional embeddings
        d_emb = self.embed(dec_input.long()) * np.sqrt(self.embed.embedding_dim)

        # apply the (stacked) decoder transformer
        decoded = self.TF_dec(d_emb + self.pos_emb[:dec_input.size(1), :],
                              dec_mask = d_mask, enc_mask = e_mask, enc_input = encoded)

        # apply the linear classification layer to the decoder output
        out = self.linear(decoded)

        return out, targs
    
    # function to generate translations (i.e. do not allow the decoder to see
    # any part of the translated sentence, also works if no translation is yet availlable)
    # works only if batch size is set to 1 in the test loop.
    def encoder_decoder_test(self, enc_input, dec_input = None, 
                             dtype = torch.cuda.FloatTensor, max_len = 64, beam_width = 1):
        # create the targets if dec_input is given, decoder input is only used
        # to create targets (e.g. for calculating a loss or comparing translation to golden standard)
        if not dec_input is None:
            targs = torch.nn.functional.pad(dec_input[:, 1:], [0, max_len - dec_input[:, 1:].size()[-1]]).long()
        else:
            targs = dtype([0])
        # create the encoder mask which is 0 where the input is padded along the time dimension
        e_mask = self.create_enc_mask(enc_input)
        # retrieve embeddings for the sentence and scale the embeddings importance relative to the pos embeddings
        emb = self.embed(enc_input.long()) * np.sqrt(self.embed.embedding_dim)
        # apply the (stacked) encoder transformer
        encoded = self.TF_enc(emb + self.pos_emb[:enc_input.size(1), :], mask = e_mask)  
        # set the decoder input to the <bos> token (i.e. predict the tranlation using only
        # the encoder output and a <bos> token)      
        dec_input = enc_input[:,0:1]
        # create the initial candidate consisting of <bos> and (negative) prob 1
        candidates = [[dec_input, 0]]
        # perform beam search
        for x in range(1, max_len + 1):
            candidates = self.beam_search(candidates, encoded, e_mask, beam_width, dtype)
        # create label predictions for the top candidate (e.g. to calculate a cross
        # entropy loss)
        d_mask = self.create_dec_mask(candidates[0][0][:, :-1])
        # convert data to embeddings
        emb = self.embed(candidates[0][0].long()) * np.sqrt(self.embed.embedding_dim)
        # pass the data through the decoder
        decoded = self.TF_dec(emb[:, :-1, :] + self.pos_emb[:candidates[0][0].size(1), :], dec_mask = d_mask,
                              enc_mask = e_mask, enc_input = encoded)
        top_pred = self.linear(decoded)
        return candidates, top_pred, targs   
    
    # beam search algorithm for finding the top n translations
    def beam_search(self, candidates, encoded, e_mask, beam_width, dtype):
        new_candidates = []
        for input, score in candidates:
            # create the decoder mask
            d_mask = self.create_dec_mask(input)
            # convert data to embeddings
            emb = self.embed(input.long()) * np.sqrt(self.embed.embedding_dim)
            # pass the data through the decoder
            decoded = self.TF_dec(emb + self.pos_emb[:input.size(1), :], dec_mask = d_mask,
                                  enc_mask = e_mask, enc_input = encoded)
            # pass the data through the prediction layer
            pred = torch.nn.functional.softmax(self.linear(decoded), dim = -1).squeeze(0)
            # get the top k predictions for the next word, calculate the new
            # probability of the sentence and append to the list of new candidates
            for value, idx in zip(*pred[-1, :].cpu().data.topk(beam_width)):
                new_candidates.append([torch.cat([input, dtype([idx.item()]).unsqueeze(0)], 1), -np.log(value) + score])
        sorted(new_candidates, key=lambda s: s[1])
        return new_candidates[:beam_width]
    
    # Function used for a transformer with an encoder only without additional context input
    # e.g. for next word prediction
    def encoder_only(self, enc_input):
        # create the targets (decoder input, shifted to the left, padded with a zero)
        targs = torch.nn.functional.pad(enc_input[:,1:], [0,1]).long()
        
        # create a mask for the padding, which also prevents the encoder from looking into the future.
        e_mask = self.create_dec_mask(enc_input)
        
        # retrieve embeddings for the sentence and scale the embeddings importance relative to the pos embeddings
        e_emb = self.embed(enc_input.long()) * np.sqrt(self.embed.embedding_dim)
        
        # apply the (stacked) encoder transformer
        encoded = self.TF_enc(e_emb + self.pos_emb[:enc_input.size(1), :], mask = e_mask)
        
        # apply the linear layer to the last decoder output (we don't want to use
        # the final version of pred because this can't be used in a cross entropy loss function)
        out = self.linear(encoded)
        return out, targs        
