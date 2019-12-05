#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 17:54:07 2018
This script contains several encoder architectures usable for the next
word prediction task. 
@author: danny
"""

from costum_layers import (multi_attention, transformer_encoder, 
transformer_att, transformer)
from load_embeddings import load_word_embeddings

import torch.nn as nn
import logging

log = logging.getLogger(__name__)
################################### transformer architectures #########################################
# the nwp task requires only the encoder side of the transformer architecture
# I made costum forward functions for the transformer that allow for training
# in an encoder-decoder setup, beam-search prediction setup, and training
# in and encoder-only setup. 

# transformer used for next word prediction ending in the same double 
# linear prediction layer as the rnn's in Aurnhammer et al. Tested this
# just to be sure but it doesn't make any difference
class nwp_transformer_2lin(transformer):
    def __init__(self, config, log = True):
        super(nwp_transformer_2lin, self).__init__()
        embed = config['embed']
        tf= config['tf']

        self.is_cuda = config['cuda']
        self.max_len = tf['max_len']

        self.embed = nn.Embedding(num_embeddings = embed['n_embeddings'], 
                                  embedding_dim = embed['embedding_dim'], 
                                  sparse = embed['sparse'],
                                  padding_idx = embed['padding_idx'])
        # prepares the positional embeddings
        self.pos_emb = self.pos_embedding(tf['max_len'],embed['embedding_dim'])

        self.TF_enc = transformer_encoder(in_size = tf['in_size'], 
                                          fc_size = tf['fc_size'], 
                                          n_layers = tf['n_layers'], 
                                          h = tf['h'])

        self.linear = nn.Sequential(nn.Linear(tf['in_size'], 
                                              tf['in_size']
                                              ), 
                                    nn.Tanh(), 
                                    nn.Linear(tf['in_size'], 
                                              embed['n_embeddings']
                                              )
                                    )
        if log:
            self.log(embed, tf)
            
    # l is included as a dummy to keep all code compatible with both 
    # Transformers and RNNs (required for pack_padded_sequence)
    def forward(self, input, l = False):

        out, targs = self.encoder_train(input)
        # apply the classification layer to the transformer output
        out = self.linear(out)        
        return out, targs
    
    def log(self, embed, tf):
        log.info('Using the standard transformer encoder with a multi-layer classification layer')
        log.info('Embedding layer: %s\nTransformer layer: %s', 
                 embed, tf)
        
# Vannilla next word prediction transformer 
class nwp_transformer(transformer):
    def __init__(self, config, log = True):
        super(nwp_transformer, self).__init__()
        
        embed = config['embed']
        tf= config['tf']

        self.is_cuda = config['cuda']
        self.max_len = tf['max_len']

        self.embed = nn.Embedding(num_embeddings = embed['n_embeddings'], 
                                  embedding_dim = embed['embedding_dim'], 
                                  sparse = embed['sparse'],
                                  padding_idx = embed['padding_idx'])
        # prepares the positional embeddings
        self.pos_emb = self.pos_embedding(tf['max_len'], embed['embedding_dim'])

        self.TF_enc = transformer_encoder(in_size = tf['in_size'], 
                                          fc_size = tf['fc_size'], 
                                          n_layers = tf['n_layers'], 
                                          h = tf['h'])
        # linear layer has no extra configurations, it just maps directly
        # from the transformer output to the number of embeddings
        self.linear = nn.Linear(tf['in_size'], embed['n_embeddings'])
        
        if log:
            self.log(embed, tf)
            
    # encoder_only packs all the transformer actions together for an 
    # encoder-only setup
    def forward(self, input, l = False):

        out, targs = self.encoder_train(input)
        # apply the classification layer to the transformer output
        out = self.linear(out)
        return out, targs

    def log(self, embed, tf):
        log.info('Using the standard transformer encoder but with decoder (future) masking')
        log.info('Embedding layer: %s\nTransformer layer: %s', 
                 embed, tf)
        
########################################################################################################

# RNN used for next word prediction with attention in between the RNN and 
# linear classification layer. Uses vectorial perceptron self attention
class nwp_rnn_att(nn.Module):
    def __init__(self, config, log = True):
        super(nwp_rnn_att, self).__init__()
        
        embed = config['embed']
        rnn = config['rnn']
        lin = config['lin']
        att = config ['att']

        self.max_len = config['max_len']
        self.embed = nn.Embedding(num_embeddings = embed['n_embeddings'], 
                                  embedding_dim = embed['embedding_dim'], 
                                  sparse = embed['sparse'],
                                  padding_idx = embed['padding_idx'])

        self.RNN = nn.GRU(input_size = rnn['in_size'], 
                          hidden_size = rnn['hidden_size'], 
                          num_layers = rnn['n_layers'], 
                          batch_first = rnn['batch_first'],
                          bidirectional = rnn['bidirectional'], 
                          dropout = rnn['dropout'])

        self.att = multi_attention(in_size = att['in_size'], 
                                   hidden_size = att['hidden_size'], 
                                   n_heads = att['h'])
        
        self.linear = nn.Sequential(nn.Linear(rnn['hidden_size'], 
                                              lin['hidden_size']
                                              ), 
                                    nn.Tanh(), 
                                    nn.Linear(lin['hidden_size'],
                                              embed['n_embeddings']
                                              )
                                    )
        if log:
            self.log(embed, rnn, lin, att)
            
    def forward(self, input, sent_lens):
        # create the targets by shifting the input left
        targs = nn.functional.pad(input[:,1:], [0,1]).long()

        embeddings = self.embed(input.long())

        x = nn.utils.rnn.pack_padded_sequence(embeddings, sent_lens, 
                                              batch_first = True, 
                                              enforce_sorted = False)
        x, hx = self.RNN(x)
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)
        
        out = self.linear(self.att(x))
        
        return out, targs
    
    def load_embeddings(self, dict_loc, embedding_loc):
        # optionally load pretrained word embeddings. takes the dictionary of 
        # words in the training data and the location of the embeddings
        load_word_embeddings(dict_loc, embedding_loc, self.embed.weight.data)     

    def log(self, embed, rnn, lin, att):
        log.info('Using the rnn (GRU) encoder with vectorial self attention')
        log.info('Embedding layer: %s\nGRU layer: %s\nLinear layer: %s\nAttention: %s', 
                 embed, rnn, lin, att)

# RNN encoder with transformer like self attention in between the RNN layer 
# and the linear classification layer. 
class nwp_rnn_tf_att(transformer):
    def __init__(self, config, log = True):
        super(nwp_rnn_tf_att, self).__init__()
        
        embed = config['embed']
        rnn = config['rnn']
        lin = config['lin']
        att = config['att']
        # is_cuda is required for the tranformer attention to create a mask of
        # the proper datatype
        self.is_cuda = config['cuda']
        self.max_len = config['max_len']
        
        self.embed = nn.Embedding(num_embeddings = embed['n_embeddings'], 
                                  embedding_dim = embed['embedding_dim'], 
                                  sparse = embed['sparse'],
                                  padding_idx = embed['padding_idx'])

        self.RNN = nn.GRU(input_size = rnn['in_size'], 
                          hidden_size = rnn['hidden_size'], 
                          num_layers = rnn['n_layers'], 
                          batch_first = rnn['batch_first'],
                          bidirectional = rnn['bidirectional'], 
                          dropout = rnn['dropout'])

        self.att = transformer_att(in_size = att['in_size'], 
                                   h = att['heads'])

        self.linear = nn.Sequential(nn.Linear(rnn['hidden_size'], 
                                              lin['hidden_size']
                                              ), 
                                    nn.Tanh(), 
                                    nn.Linear(lin['hidden_size'],
                                              embed['n_embeddings']
                                              )
                                    )
        if log:
            self.log(embed, rnn, lin, att)
            
    def forward(self, input, sent_lens):
        # create the targets by shifting the input left
        targs = nn.functional.pad(input[:,1:], [0,1]).long()
        # create a mask for the attention, preventing the net from peeking
        # at the future
        mask = self.create_dec_mask(input)

        embeddings = self.embed(input.long())

        x = nn.utils.rnn.pack_padded_sequence(embeddings, sent_lens, 
                                              batch_first = True,
                                              enforce_sorted = False)
        x, hx = self.RNN(x)
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)
        
        out = self.linear(self.att(q = x, k = x, v = x, mask = mask))
        
        return out, targs
    
    def load_embeddings(self, dict_loc, embedding_loc):
        # optionally load pretrained word embeddings. takes the dictionary of 
        # words in the training data and the location of the embeddings
        load_word_embeddings(dict_loc, embedding_loc, self.embed.weight.data)     

    def log(self, embed, rnn, lin, att):
        log.info('Using the rnn (GRU) encoder with transformer like self attention')
        log.info('Embedding layer: %s\nGRU layer: %s\nLinear layer: %s\nAttention: %s', 
                 embed, rnn, lin, att)

# Vanilla RNN used for next word prediction, no attention
class nwp_rnn_encoder(nn.Module):
    def __init__(self, config, log = True):
        super(nwp_rnn_encoder, self).__init__()
        embed = config['embed']
        rnn = config['rnn']
        lin = config['lin']

        self.max_len = config['max_len']
        
        self.embed = nn.Embedding(num_embeddings = embed['n_embeddings'], 
                                  embedding_dim = embed['embedding_dim'], 
                                  sparse = embed['sparse'],
                                  padding_idx = embed['padding_idx'])

        self.RNN = nn.GRU(input_size = rnn['in_size'], 
                          hidden_size = rnn['hidden_size'], 
                          num_layers = rnn['n_layers'],
                          batch_first = rnn['batch_first'],
                          bidirectional = rnn['bidirectional'], 
                          dropout = rnn['dropout'])

        self.linear = nn.Sequential(nn.Linear(rnn['hidden_size'], 
                                              lin['hidden_size']
                                              ), 
                                    nn.Tanh(), 
                                    nn.Linear(lin['hidden_size'],
                                              embed['n_embeddings']
                                              )
                                    )
        if log:
            self.log(embed, rnn, lin)
            
    def forward(self, input, sent_lens):
        # create the targets by shifting the input left
        targs = nn.functional.pad(input[:,1:], [0,1]).long()

        embeddings = self.embed(input.long())

        x = nn.utils.rnn.pack_padded_sequence(embeddings, sent_lens, 
                                              batch_first = True,
                                              enforce_sorted = False)
        x, hx = self.RNN(x)
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)

        out = self.linear(x)  
        
        return out, targs
    
    def load_embeddings(self, dict_loc, embedding_loc):
        # optionally load pretrained word embeddings. takes the dictionary of 
        # words in the training data and the location of the embeddings
        load_word_embeddings(dict_loc, embedding_loc, self.embed.weight.data) 

    def log(self, embed, rnn, lin):
        log.info('Using the standard rnn (GRU) encoder')
        log.info('Embedding layer: %s\nGRU layer: %s\nLinear layer: %s', 
                 embed, rnn, lin)
