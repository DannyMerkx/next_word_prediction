#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:36:50 2019

@author: danny
"""
import torch
import sys
import pickle
import numpy as np
import pandas as pd
import os
# make sure this script is in the same folder as the functions folder of the nwp project
sys.path.append('/home/danny/Documents/project code/next_word_prediction/PyTorch/functions')

from collections import defaultdict
from encoders import nwp_transformer, nwp_rnn_encoder
from prep_text import word_2_index
# location of a pre-trained model
model_loc = '/home/danny/Downloads/models'
# location of the sentences to be encoded.
data_loc = '/home/danny/Documents/databases/next_word_prediction/test.txt'
dict_loc = './nwp_indices'

# list all the pretrained models
model_list = [x for x in os.walk(model_loc)]
model_list = [os.path.join(x[0], y) for x in model_list[1:] for y in x[2]]
model_list.sort()
# function to load a pickled dictionary
def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)
# load the location of dictionary indices
nwp_dict = load_obj(dict_loc)
dict_size = len(nwp_dict) + 1 

# function to calculate the average loss over your test data
def test_loss(data_loc, model, loss):
    l = 0
    it = 0
    with open(data_loc) as file:
        for line in file:
            # split the sentence into tokens
            sent = [line.split()]
            # convert text to indices
            prep_text, s_lengths = word_2_index(sent, 1, 'nwp_indices')
            # get the predictions and targets for this sentence
            prediction, targets = nwp_model(torch.FloatTensor(prep_text), s_lengths)
    
            
            l += loss(prediction.view(-1, prediction.size(-1)), targets.view(-1)).data
            it += 1
    print(l/it)

def index_2_word(dictionary, indices):
    rev_dictionary = defaultdict(str)
    for x, y in dictionary.items():
        rev_dictionary[y] = x
    sentences = [[rev_dictionary[int(i)] for i in ind] for ind in indices]   
    return(sentences)

# function to sort senteces by decreasing length (necessary to use pack_padded_sequence
# in pytorch)
def sort_decrease(sents, lengths):
    # we need to keep the original argsort indices in order to reverse the sorting later  
    sort_idx = np.argsort(- np.array(lengths))
    sents = sents[sort_idx]    
    lengths = np.array(lengths)[sort_idx] 
    return sort_idx, sents, lengths
  
# function to produce the surprisal ratings for your test data
def calc_surprisal(data_loc, model, loss):
    # list to hold all the surprisal values
    surprisal = []
    # open the database
    sent = []
    with open(data_loc) as file:
        for line in file:
            # split the sentence into tokens
            sent.append(['<s>'] + line.split() + ['</s>'])
    # convert text to indices, keep original sentence length to remove padding later
    sents, sent_l = word_2_index(sent, 361, 'nwp_indices')
    # sort the sentences by decreasing length (needed for pack_padded_sequence in the gru)
    sort_idx, sents, sent_l = sort_decrease(sents, sent_l)
    # get the predictions and targets for this sentence
    predictions, targets = model(torch.FloatTensor(sents), sent_l)
    # reverse the sorting
    predictions = predictions[np.argsort(sort_idx)]
    targets = targets[np.argsort(sort_idx)]
    sent_l = sent_l[np.argsort(sort_idx)]
    # convert the predictions to surprisal (negative log softmax)
    surprisal = -torch.log_softmax(predictions, dim = 2).squeeze()
    # extract only the surpisal ratings for the target words
    surprisal = [[float(surprisal[x_index][y_index, t]) for y_index, t in enumerate(targ)] for x_index, targ in enumerate(targets) ]
    # now remove any padding applied by word_2_index and remove end of sentence prediction
    surprisal = [s[:l -2] for s, l in zip(surprisal, sent_l)] 
    l = loss(predictions.view(-1, predictions.size(-1)), targets.view(-1)).data  
    return(surprisal, targets, l)

# set contractions (words with ' in it) and words followed by a comma to nan as
# they were excluded in the original experiment. e.g. "would n't" and "did ," are two separate 
# tokens in our training/test data but they were presented as one token to the participants.
# In our surprisal data we want to count such token pairs as one token and set their value to nan.
# This is done by finding tokens with punctuation in them, set the surprisal of the previous token
# to nan and delete the token itself. 
def clean_surprisal(surprisal, targets):
    for s, t in zip(surprisal, targets):
        # iterate over all test data to find occurences of punctuation in the targets
        for index, word in enumerate(t):
            if '\'' in word or ',' in word:
                # set the previous token to nan
                s[index-1] = np.nan
    # construct lists of which words to keep, that is set keep to false for all items
    # for which the previous value was nan. Set the first item to True by default (has no previous item)
    keep = [[True] + [not(np.isnan(s[ind-1])) for ind in range(1, len(s))] for s in surprisal]   
    # now keep only those surprisal values we need to keep 
    surprisal = [[s[x] for x in range(len(s)) if k[x]] for s, k in zip(surprisal, keep)]
    # add sentence and word position indices to the surprisal ratings and convert to DataFrame object
    surprisal = [(sent_index + 1, word_index +1, word) for sent_index, sent in enumerate(surprisal) for word_index, word in enumerate(sent)]
    surprisal = pd.DataFrame(np.array(surprisal))
    return surprisal

# config settings for the models;
transformer_config = {'embed': {'num_embeddings': dict_size,'embedding_dim': 400, 'sparse': False, 'padding_idx':0}, 
          'tf':{'input_size':400, 'fc_size': 2048,'n_layers': 2,'h': 8, 'max_len': 41},
          'cuda': False}
gru_config = {'embed':{'num_embeddings': dict_size, 'embedding_dim': 400, 'sparse': False, 'padding_idx': 0}, 'max_len': 41,
               'rnn':{'input_size': 400, 'hidden_size': 500, 'num_layers': 1, 'batch_first': True,
               'bidirectional': False, 'dropout': 0}, 'lin1':{'input_size': 500, 'output_size': 400}, 'lin2':{'input_size': 400}}

# set the crossentropy loss function
loss = torch.nn.CrossEntropyLoss(ignore_index = 0)
# create the models
transformer_model = nwp_transformer(transformer_config)
gru_model = nwp_rnn_encoder(gru_config)
###############################################################################
data = pd.DataFrame()
# variable to store all the loss values (i.e. language model performance)
loss_vals = []
for model_loc in model_list:
    # load the pretrained model
    model = torch.load(model_loc, map_location = 'cpu')
    if 'gru' in model_loc:
        nwp_model = gru_model
    else:
        nwp_model = transformer_model
    # load the pretrained weights in the model
    nwp_model.load_state_dict(model)
    # set requires grad to false for faster encoding
    for param in nwp_model.parameters():
        param.requires_grad = False
    # set to eval mode to disable dropout
    nwp_model.eval()
    # load the dictionary of indexes and create a reverse lookup dictionary so
    # we can look up target words by their index
    index_dict = load_obj(dict_loc)
    word_dict = defaultdict(str)
    for x, y in index_dict.items():
        word_dict[y] = x
    # get all the surprisal values and the target sequence (inputs shifted to the left)
    surprisal, targets, l = calc_surprisal(data_loc, nwp_model, loss)
    
    # convert the target indices back to words
    targets = index_2_word(nwp_dict, targets)

    surprisal = clean_surprisal(surprisal, targets)
    # create a unique name for the current surprisal values by combining the model number
    # with the nr of training samples of the model
    surp_name = model_loc.split('/')[-2] + '_' + model_loc.split('/')[-1].split('.')[-1]
    surprisal.columns = ['sent_nr', 'word_pos', surp_name ]
    loss_vals.append([l.numpy(), surp_name])
    item_nr = []
    for x, y in zip(surprisal.sent_nr, surprisal.word_pos):    
        x = x*100
        item_nr.append(int(x+y))
    surprisal['item'] = pd.Series(item_nr)
    if not data.empty:
        data[surp_name] = data.join(surprisal[[surp_name, 'item']].set_index('item'), on = 'item')[surp_name]
    else:
        data = surprisal
###############################################################################
# now sort the column names in the surprisal data
col_names = data.columns.tolist()
models = col_names[2:3] + col_names[4:]
models.sort()
col_names = col_names[0:2] + col_names[3:4] + models
data = data[col_names]
# round the surprisal to 4 decimals and convert the sent_nr and word_pos from float to in
data[models] = data[models].round(4)
data.sent_nr = data.sent_nr.astype(int)
data.word_pos = data.word_pos.astype(int)

data.to_csv(path_or_buf = '/home/danny/Downloads/surprisal.csv')
loss_vals = pd.DataFrame(loss_vals)
loss_vals.columns = ['loss', 'model_name']
loss_vals.loss = loss_vals.loss.astype(float)
loss_vals.to_csv(path_or_buf = '/home/danny/Downloads/lm_accuracy.csv')