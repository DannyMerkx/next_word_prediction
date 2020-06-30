#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 16:12:05 2018
load pretrained word embeddings (GloVe) and use them to initialise an embedding layer
@author: danny
"""
import pickle
import torch
from collections import defaultdict
# load dictionary
def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)
# make a dictionary of the glove vectors for words occuring in the training data
def make_glove_dict(glove, index_dict):
    glove_dict = defaultdict(str)
    for line in glove:
        line = line.split(' ')
        # if the word occurs in our data we add the glove vector to the dictionary
        if index_dict[line[0]] != 0:
            glove_dict[line[0]] = line[1:] 
    return glove_dict
        
def load_word_embeddings(dict_loc, embedding_loc, embeddings):  
    # load the dictionary containing the indices of the words in the training data
    index_dict = load_obj(dict_loc)
    # load the file with the pretraind glove embeddings
    glove = open(embedding_loc)
    # make the dictionary of words in the training data that have a glove vector
    glove_dict = make_glove_dict(glove, index_dict)
    # print for how many words we could load pretrained vectors
    print('found ' + str(len(glove_dict)) + ' glove vectors')
    index_dict = load_obj(dict_loc)
    # replace the random embeddings with the pretrained embeddings
    for key in glove_dict.keys():
        index = index_dict[key]
        if index == 0:
            print('found a glove vector that does not occur in the data')
        emb = torch.FloatTensor([float(x) for x in glove_dict[key]])
        embeddings[index] = emb