#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 17:54:07 2018

@author: danny
"""

from costum_layers import multi_attention, transformer_encoder, transformer_decoder, transformer_att
from load_embeddings import load_word_embeddings
from transformer import transformer

import torch
import torch.nn as nn

################################### transformer architectures #########################################

# transformer used for next word prediction. Uses only the encoder part of the original transformer. 
class nwp_transformer(transformer):
    def __init__(self, config):
        super(nwp_transformer, self).__init__()
        embed = config['embed']
        tf= config['tf']
        lin = config['lin']
        # makes sure variables are mapped to the gpu if needed
        self.is_cuda = config['cuda']
        self.max_len = tf['max_len']
        # create the embedding layer
        self.embed = nn.Embedding(num_embeddings = embed['num_embeddings'], 
                                  embedding_dim = embed['embedding_dim'], sparse = embed['sparse'],
                                  padding_idx = embed['padding_idx'])
        # create the positional embeddings
        self.pos_emb = self.pos_embedding(tf['max_len'],embed['embedding_dim'])
        # create the (stacked) transformer
        self.TF_enc = transformer_encoder(in_size = tf['input_size'], fc_size = tf['fc_size'], 
                              n_layers = tf['n_layers'], h = tf['h'])
        # linear layer maps to the output dictionary
	self.linear = nn.Sequential(nn.Linear(tf['input_size'], tf['input_size']), nn.Tanh(), 
                                    nn.Linear(tf['input_size'], embed['num_embeddings']))

    # l is included as a dummy to keep all code compatible with both Transformers and RNNs (which need the length of the unpadded sentences)
    def forward(self, input, l = False):
        # encode the sentence using the transformer. encoder_only uses the encoder part only but
        # with a decoder mask (which prevents peeking at future timesteps)
        out, targs = self.encoder_only(input)
        return out, targs

# transformer used for next word prediction. Uses only the encoder part of the original transformer. 
class nwp_transformer_simple(transformer):
    def __init__(self, config):
        super(nwp_transformer, self).__init__()
        embed = config['embed']
        tf= config['tf']
        # makes sure variables are mapped to the gpu if needed
        self.is_cuda = config['cuda']
        self.max_len = tf['max_len']
        # create the embedding layer
        self.embed = nn.Embedding(num_embeddings = embed['num_embeddings'], 
                                  embedding_dim = embed['embedding_dim'], sparse = embed['sparse'],
                                  padding_idx = embed['padding_idx'])
        # create the positional embeddings
        self.pos_emb = self.pos_embedding(tf['max_len'],embed['embedding_dim'])
        # create the (stacked) transformer
        self.TF_enc = transformer_encoder(in_size = tf['input_size'], fc_size = tf['fc_size'], 
                              n_layers = tf['n_layers'], h = tf['h'])
        # linear layer maps to the output dictionary
        self.linear = nn.Linear(tf['input_size'], embed['num_embeddings'])
    # l is included as a dummy to keep all code compatible with both Transformers and RNNs (which need the length of the unpadded sentences)
    def forward(self, input, l = False):
        # encode the sentence using the transformer. encoder_only uses the encoder part only but
        # with a decoder mask (which prevents peeking at future timesteps)
        out, targs = self.encoder_only(input)
        return out, targs

########################################################################################################

# RNN used for next word prediction with attention in between the RNN and linear classification layer
class nwp_rnn_att(nn.Module):
    def __init__(self, config):
        super(nwp_rnn_encoder, self).__init__()
        embed = config['embed']
        rnn = config['rnn']
        lin_1 = config['lin1']
        lin_2 = config['lin2']
        att = config ['att']

        self.max_len = config['max_len']
        self.embed = nn.Embedding(num_embeddings = embed['num_embeddings'], 
                                  embedding_dim = embed['embedding_dim'], sparse = embed['sparse'],
                                  padding_idx = embed['padding_idx'])

        self.RNN = nn.GRU(input_size = rnn['input_size'], hidden_size = rnn['hidden_size'], 
                          num_layers = rnn['num_layers'], batch_first = rnn['batch_first'],
                          bidirectional = rnn['bidirectional'], dropout = rnn['dropout'])

        self.att = multi_attention(in_size = rnn['hidden_size'], hidden_size = att['hidden_size'], n_heads = att['heads'])
        
        self.linear = nn.Sequential(nn.Linear(rnn['hidden_size'], lin['output_size']), nn.Tanh(), nn.Linear(lin['output_size'],
                                    embed['num_embeddings']))

    def forward(self, input, sent_lens):
	# create the targets by shifting the input left
        targs = torch.nn.functional.pad(input[:,1:], [0,1]).long()

        embeddings = self.embed(input.long())
        # create a packed_sequence object. The padding will be excluded from the update step
        # thereby training on the original sequence length only
        x = torch.nn.utils.rnn.pack_padded_sequence(embeddings, sent_lens, batch_first = True)

        x, hx = self.RNN(x)
        # unpack again as at the moment only rnn layers except packed_sequence objects
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)
	# use the linear layers to map to the output dictionary
        x = self.linear(self.att(x))  
        return x, targs
    
    def load_embeddings(self, dict_loc, embedding_loc):
        # optionally load pretrained word embeddings. takes the dictionary of words occuring in the training data
        # and the location of the embeddings.
        load_word_embeddings(dict_loc, embedding_loc, self.embed.weight.data)     

# RNN encoder with transformer like self attention in between the RNN layer and the linear classification layer. 
class nwp_rnn_tf_att(nn.Module):
    def __init__(self, config):
        super(nwp_rnn_encoder, self).__init__()
        embed = config['embed']
        rnn = config['rnn']
        lin_1 = config['lin1']
        lin_2 = config['lin2']
        att = config ['att']

        self.max_len = config['max_len']
        self.embed = nn.Embedding(num_embeddings = embed['num_embeddings'], 
                                  embedding_dim = embed['embedding_dim'], sparse = embed['sparse'],
                                  padding_idx = embed['padding_idx'])

        self.RNN = nn.GRU(input_size = rnn['input_size'], hidden_size = rnn['hidden_size'], 
                          num_layers = rnn['num_layers'], batch_first = rnn['batch_first'],
                          bidirectional = rnn['bidirectional'], dropout = rnn['dropout'])

        self.att = transformer_att(in_size = att['in_size'], h = att['heads'])

        self.linear = nn.Sequential(nn.Linear(rnn['hidden_size'], lin['output_size']), nn.Tanh(), nn.Linear(lin['output_size'],
                                    embed['num_embeddings']))

    def forward(self, input, sent_lens):
	# create the targets by shifting the input left
        targs = torch.nn.functional.pad(input[:,1:], [0,1]).long()

        embeddings = self.embed(input.long())
        # create a packed_sequence object. The padding will be excluded from the update step
        # thereby training on the original sequence length only
        x = torch.nn.utils.rnn.pack_padded_sequence(embeddings, sent_lens, batch_first = True)

        x, hx = self.RNN(x)
        # unpack again as at the moment only rnn layers except packed_sequence objects
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)
        x = self.att(x)
	# use the linear layers to map to the output dictionary
        x = self.linear(self.att(x))  
        return x, targs
    
    def load_embeddings(self, dict_loc, embedding_loc):
        # optionally load pretrained word embeddings. takes the dictionary of words occuring in the training data
        # and the location of the embeddings.
        load_word_embeddings(dict_loc, embedding_loc, self.embed.weight.data)     


# Vanilla RNN used for next word prediction
class nwp_rnn_encoder(nn.Module):
    def __init__(self, config):
        super(nwp_rnn_encoder, self).__init__()
        embed = config['embed']
        rnn = config['rnn']
        lin_1 = config['lin1']
        lin_2 = config['lin2']
        att = config ['att']

        self.max_len = config['max_len']
        self.embed = nn.Embedding(num_embeddings = embed['num_embeddings'], 
                                  embedding_dim = embed['embedding_dim'], sparse = embed['sparse'],
                                  padding_idx = embed['padding_idx'])

        self.RNN = nn.GRU(input_size = rnn['input_size'], hidden_size = rnn['hidden_size'], 
                          num_layers = rnn['num_layers'], batch_first = rnn['batch_first'],
                          bidirectional = rnn['bidirectional'], dropout = rnn['dropout'])

        self.linear = nn.Sequential(nn.Linear(rnn['hidden_size'], lin['output_size']), nn.Tanh(), nn.Linear(lin['output_size'],
                                    embed['num_embeddings']))


    def forward(self, input, sent_lens):
	# create the targets by shifting the input left
        targs = torch.nn.functional.pad(input[:,1:], [0,1]).long()

        embeddings = self.embed(input.long())
        # create a packed_sequence object. The padding will be excluded from the update step
        # thereby training on the original sequence length only
        x = torch.nn.utils.rnn.pack_padded_sequence(embeddings, sent_lens, batch_first = True)

        x, hx = self.RNN(x)
        # unpack again as at the moment only rnn layers except packed_sequence objects
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)
	# use the linear layers to map to the output dictionary
        x = self.linear(x)  
        return x, targs
    
    def load_embeddings(self, dict_loc, embedding_loc):
        # optionally load pretrained word embeddings. takes the dictionary of words occuring in the training data
        # and the location of the embeddings.
        load_word_embeddings(dict_loc, embedding_loc, self.embed.weight.data)     


