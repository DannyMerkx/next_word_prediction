#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:36:50 2019
This script loads the trained language models and uses them to extract 
surprisal values for the human reading data.
@author: danny
"""
import torch
import sys
import pickle
import numpy as np
import pandas as pd
import os
# make sure this script is in the same folder as the functions folder of the 
# nwp project
sys.path.append('../functions')

from collections import defaultdict
from encoders import *
from prep_text import word_2_index
# location of a pre-trained model
model_loc = '/home/danny/Documents/papers/COLING_paper/COLING_models'
# location of the sentences to be encoded.
data_loc = '/home/danny/Documents/databases/next_word_prediction/data/test.txt'
dict_loc = '/home/danny/Documents/databases/next_word_prediction/data/train_indices'

# list all the pretrained models
model_list = [x for x in os.walk(model_loc)]
model_list = [os.path.join(x[0], y) for x in model_list[1:] for y in x[2] if not '.out' in y]
model_list.sort()

# function to load a pickled dictionary with the indices of each possible token
def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)

nwp_dict = load_obj(dict_loc)
dict_size = len(nwp_dict) + 1 

def index_2_word(dictionary, indices):
    rev_dictionary = defaultdict(str)
    for x, y in dictionary.items():
        rev_dictionary[y] = x
    sentences = [[rev_dictionary[int(i)] for i in ind] for ind in indices]   
    return(sentences)
  
# function to produce the surprisal ratings for the test sentences
def calc_surprisal(data_loc, model):

    sent = []
    with open(data_loc) as file:
        for line in file:
            # split the sentence into tokens
            sent.append(['<s>'] + line.split() + ['</s>'])
    # convert text to indices, 
    sent, l = word_2_index(sent, len(sent), nwp_dict)

    # get the predictions and targets for this sentence
    predictions, targets = model(torch.FloatTensor(sent), l)

    # convert the predictions to surprisal (negative log softmax)
    surprisal = -torch.log_softmax(predictions, dim = 2).squeeze()
    # extract only the surpisal ratings for the target words
    surprisal = surprisal.gather(-1, targets.unsqueeze(-1)).squeeze()
    # finally remove any padding applied by word_2_index and remove end of 
    # sentence prediction
    surprisal = surprisal.data.numpy()
    surprisal = [s[:l - 2] for s, l in zip(surprisal, l)]
    return(surprisal, targets)

# set words followed by a comma to nan as they were excluded in the original 
# experiment. 
def clean_surprisal(surprisal, targets):
    for s, t in zip(surprisal, targets):
        # iterate over all test data to find occurences of punctuation in the 
        #targets
        for index, word in enumerate(t):
            if ',' in word:
                # set the previous token to nan
                s[index-1] = np.nan
    # construct lists of which words to keep, that is set keep to false for 
    # all items for which the previous value was nan. Set the first item to 
    # True by default (has no previous item)
    keep = [[True] + [not(np.isnan(s[ind-1])) for ind in range(1, len(s))] for s in surprisal]   
    # now keep only those surprisal values we need to keep 
    surprisal = [[s[x] for x in range(len(s)) if k[x]] for s, k in zip(surprisal, keep)]
    # add sentence and word position indices to the surprisal ratings and 
    # convert to DataFrame object
    surprisal = [(sent_index + 1, word_index +1, word) for sent_index, 
                 sent in enumerate(surprisal) for word_index, 
                 word in enumerate(sent)
                 ]
    surprisal = pd.DataFrame(np.array(surprisal))
    return surprisal

# config settings for the models;
tf_1l_config = {'embed': {'n_embeddings': dict_size,
                          'embedding_dim': 400, 'sparse': False, 
                          'padding_idx':0
                          }, 
                'tf':{'in_size': 400, 'fc_size': 1024,'n_layers': 1,
                      'h': 8, 'max_len': 41
                      },
                'cuda': False
                }
tf_2l_config = {'embed': {'n_embeddings': dict_size,
                                'embedding_dim': 400, 'sparse': False, 
                                'padding_idx':0
                                }, 
                      'tf':{'in_size': 400, 'fc_size': 1024,'n_layers': 2,
                            'h': 8, 'max_len': 41
                            },
                      'cuda': False
                      }

tf_4l_config = {'embed': {'n_embeddings': dict_size,
                                'embedding_dim': 400, 'sparse': False, 
                                'padding_idx':0
                                }, 
                      'tf':{'in_size': 400, 'fc_size': 1024,'n_layers': 4,
                            'h': 8, 'max_len': 41
                            },
                      'cuda': False
                      }

gru_1l_config = {'embed':{'n_embeddings': dict_size, 'embedding_dim': 400, 
                       'sparse': False, 'padding_idx': 0
                       }, 
                      'max_len': 41,
                      'rnn':{'in_size': 400, 'hidden_size': 500, 
                             'n_layers': 1, 'batch_first': True,
                             'bidirectional': False, 'dropout': 0
                             }, 
                      'lin':{'hidden_size': 400
                             }, 
                      'att': {'in_size': 500, 'heads':10
                              },
                      'cuda': False
              }

gru_2l_config = {'embed':{'n_embeddings': dict_size, 'embedding_dim': 400, 
                       'sparse': False, 'padding_idx': 0
                       }, 
                      'max_len': 41,
                      'rnn':{'in_size': 400, 'hidden_size': 500, 
                             'n_layers': 2, 'batch_first': True,
                             'bidirectional': False, 'dropout': 0
                             }, 
                      'lin':{'hidden_size': 400
                             }, 
                      'att': {'in_size': 500, 'heads': 10
                              },
                      'cuda': False
              }

# create the models
tf_1l = nwp_transformer(tf_1l_config)
tf_2l = nwp_transformer(tf_2l_config)
tf_4l = nwp_transformer(tf_4l_config)

gru_1l = nwp_rnn_encoder(gru_1l_config)
gru_2l = nwp_rnn_encoder(gru_2l_config)
gru_tf = nwp_rnn_tf_att(gru_1l_config)

encoder_models = [tf_1l, tf_2l, tf_4l, gru_1l, gru_2l, gru_tf]
###############################################################################
data = pd.DataFrame()

for model_loc in model_list:
    # load the pretrained model
    model = torch.load(model_loc, map_location = 'cpu')
    print(model_loc)
    for i, enc in enumerate(encoder_models):
        nwp_model = enc
        try:
            nwp_model.load_state_dict(model)
            break
        except:
            continue
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
    # get all the surprisal values and the target sequence (inputs shifted to
    # the left)
    surprisal, targets = calc_surprisal(data_loc, nwp_model)
    # convert the target indices back to words
    targets = index_2_word(nwp_dict, targets)

    surprisal = clean_surprisal(surprisal, targets)
    # create a unique name for the current surprisal values by combining the 
    # model number with the nr of training samples of the model. N.B. the 
    # indices here are hard coded for my specific folder structure and file 
    # naming convention. 
    surp_name = model_loc.split('/')[-2] + '_' + model_loc.split('/')[-1].split('.')[-1]
    surprisal.columns = ['sent_nr', 'word_pos', surp_name ]

    item_nr = []
    for x, y in zip(surprisal.sent_nr, surprisal.word_pos):    
        x = x*100
        item_nr.append(int(x+y))
    surprisal['item'] = pd.Series(item_nr)
    if not data.empty:
        data[surp_name] = data.join(surprisal[[surp_name, 'item']].set_index('item'), 
                                    on = 'item')[surp_name]
    else:
        data = surprisal

###############################################################################
# now sort the column names in in loading state_dict for nwp_transformer:
col_names = data.columns.tolist()
models = col_names[2:3] + col_names[4:]
models.sort()
col_names = col_names[0:2] + col_names[3:4] + models
data = data[col_names]
# round the surprisal to 4 decimals and convert the sent_nr and word_pos from 
# float to in
data[models] = data[models].round(4)
data.sent_nr = data.sent_nr.astype(int)
data.word_pos = data.word_pos.astype(int)

data.to_csv(path_or_buf = '/home/danny/Documents/databases/next_word_prediction/surprisal_data/surprisal.csv')
