#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 17:54:07 2018

@author: danny
"""

from costum_layers import RHN, multi_attention, transformer_encoder, transformer_decoder
from load_embeddings import load_word_embeddings
from transformer import transformer

import torch
import torch.nn as nn
#################################image-sentence embedders##################################################
# rnn encoder for characters and tokens, creates sentence embeddings using self attention
class text_rnn_encoder(nn.Module):
    def __init__(self, config):
        super(text_rnn_encoder, self).__init__()
        embed = config['embed']
        rnn = config['rnn']
        att = config ['att'] 
        self.embed = nn.Embedding(num_embeddings = embed['num_embeddings'], 
                                  embedding_dim = embed['embedding_dim'], sparse = embed['sparse'],
                                  padding_idx = embed['padding_idx'])
        self.RNN = nn.GRU(input_size = rnn['input_size'], hidden_size = rnn['hidden_size'], 
                          num_layers = rnn['num_layers'], batch_first = rnn['batch_first'],
                          bidirectional = rnn['bidirectional'], dropout = rnn['dropout'])
        self.att = multi_attention(in_size = att['in_size'], hidden_size = att['hidden_size'], n_heads = att['heads'])
        
    def forward(self, input, l):
        # embedding layers expect Long tensors
        x = self.embed(input.long())
        # create a packed_sequence object. The padding will be excluded from the update step
        # thereby training on the original sequence length only
        x = torch.nn.utils.rnn.pack_padded_sequence(x, l, batch_first=True)
        x, hx = self.RNN(x)
        # unpack again as at the moment only rnn layers except packed_sequence objects
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)
        x = nn.functional.normalize(self.att(x), p=2, dim=1)    
        return x
    
    def load_embeddings(self, dict_loc, embedding_loc):
        # optionally load pretrained word embeddings. takes the dictionary of words occuring in the training data
        # and the location of the embeddings.
        load_word_embeddings(dict_loc, embedding_loc, self.embed.weight.data)         
    
# the network for embedding visual features
class img_encoder(nn.Module):
    def __init__(self, config):
        super(img_encoder, self).__init__()
        linear = config['linear']
        self.norm = config['norm']
        self.linear_transform = nn.Linear(in_features = linear['in_size'], out_features = linear['out_size'])
        nn.init.xavier_uniform(self.linear_transform.weight.data)
    def forward(self, input):
        x = self.linear_transform(input)
        if self.norm:
            return nn.functional.normalize(x, p=2, dim=1)
        else:
            return x
################################### transformer architectures #########################################

# transformer model which takes aligned input in two languages and learns
# to translate from language to the other. 
class translator_transformer(transformer):
    def __init__(self, config):
        super(translator_transformer, self).__init__()
        embed = config['embed']
        tf = config['tf']
        self.is_cuda = config['cuda']
        self.max_len = tf['max_len']
        # create the embedding layer
        self.embed = nn.Embedding(num_embeddings = embed['num_chars'], 
                                  embedding_dim = embed['embedding_dim'], sparse = embed['sparse'],
                                  padding_idx = embed['padding_idx'])
        # create the positional embeddings
        self.pos_emb = self.pos_embedding(tf['max_len'], embed['embedding_dim'])
        # create the (stacked) transformer
        self.TF_enc = transformer_encoder(in_size = tf['input_size'], fc_size = tf['fc_size'], 
                              n_layers = tf['n_layers'], h = tf['h'])
        self.TF_dec = transformer_decoder(in_size = tf['input_size'], fc_size = tf['fc_size'], 
                              n_layers = tf['n_layers'], h = tf['h'])
        self.linear = nn.Linear(embed['embedding_dim'], embed['num_chars'])
    # forward, during training give the transformer both languages
    def forward(self, enc_input, dec_input):
        out, targs = self.encoder_decoder_train(enc_input, dec_input)
        return out, targs
    # translate, during test time translate from one language to the other, works without decoder input
    # from the target language. 
    def translate(self, enc_input, dec_input = None, dtype = torch.cuda.FloatTensor, beam_width = 1):
        candidates, preds, targs = self.encoder_decoder_test(enc_input, dec_input, dtype, self.max_len, beam_width)
        return candidates, preds, targs

# transformer used for next word prediction. Uses only the encoder part of the original transformer. 
class nwp_transformer(transformer):
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

# RNN used for next word prediction
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

        self.att = multi_attention(in_size = att['in_size'], hidden_size = att['hidden_size'], n_heads = att['heads'])

        self.linear_1 = nn.Linear(lin_1['input_size'], lin_1['output_size'])       
        self.linear_2 = nn.Linear(lin_2['input_size'], embed['num_embeddings'])
        self.tan = nn.Tanh() 

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
        x = self.linear_2(self.tan(self.linear_1(x)))  
        return x, targs
    
    def load_embeddings(self, dict_loc, embedding_loc):
        # optionally load pretrained word embeddings. takes the dictionary of words occuring in the training data
        # and the location of the embeddings.
        load_word_embeddings(dict_loc, embedding_loc, self.embed.weight.data)     

######################################################################################################
# network concepts and experiments and networks by others
######################################################################################################

# simple encoder that just sums the word embeddings of the tokens
class bow_encoder(nn.Module):
    def __init__(self, config):
        super(bow_encoder, self).__init__()
        embed = config['embed']
        self.embed = nn.Embedding(num_embeddings = embed['num_chars'], 
                                  embedding_dim = embed['embedding_dim'], sparse = embed['sparse'],
                                  padding_idx = embed['padding_idx'])
    def forward(self, input, l):
        # embedding layers expect Long tensors
        x = self.embed(input.long())
        return x.sum(2)
    
    def load_embeddings(self, dict_loc, embedding_loc):
        # optionally load pretrained word embeddings. takes the dictionary of words occuring in the training data
        # and the location of the embeddings.
        load_word_embeddings(dict_loc, embedding_loc, self.embed.weight.data) 

# the convolutional character encoder described by Wehrmann et al. 
class conv_encoder(nn.Module):
    def __init__(self):
        super(conv_encoder, self).__init__()
        self.Conv1d_1 = nn.Conv1d(in_channels = 20, out_channels = 512, kernel_size = 7,
                                 stride = 1, padding = 3, groups = 1)
        self.Conv1d_2 = nn.Conv1d(in_channels = 512, out_channels = 512, kernel_size = 5,
                                 stride = 1, padding = 2, groups = 1)
        self.Conv1d_3 = nn.Conv1d(in_channels = 512, out_channels = 512, kernel_size = 3,
                                 stride = 1, padding = 1, groups = 1)
        self.relu = nn.ReLU()
        self.embed = nn.Embedding(num_embeddings = 100, embedding_dim = 20,
                                  sparse = False, padding_idx = 0)
        self.Pool = nn.AdaptiveMaxPool1d(output_size = 1, return_indices=False)
        self.linear = nn.Linear(in_features = 512, out_features = 512)
    def forward(self, input, l):
        x = self.embed(input.long()).permute(0,2,1)
        x = self.relu(self.Conv1d_1(x))
        x = self.relu(self.Conv1d_2(x))
        x = self.relu(self.Conv1d_3(x))
        x = self.linear(self.Pool(x).squeeze())
        return nn.functional.normalize(x, p = 2, dim = 1)
