 #!/u)sr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 13:56:15 2018

@author: danny

Functions to prepare text data by replacing the tokens/characters with embedding
layer indices.
"""
import string
import numpy as np
import pickle

# loader for the dictionary, loads a pickled dictionary.
def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)

def find_index(char):
    # define the set of valid characters.
    valid_chars = string.printable 
    idx = valid_chars.find(char)
    # add 1 to the index so the range is 1 to 101 and 0 is free to use as padding idx
    if idx != -1:
        idx += 1
    return idx

def char_2_index(batch, batch_size):
    max_sent_len = max([len(x) for x in batch])
    index_batch = np.zeros([batch_size, max_sent_len])
    # keep track of the origin sentence length to use in pack_padded_sequence
    lengths = []
    for i, text in enumerate(batch):
        lengths.append(len(text))        
        for j, char in enumerate(text):
            index_batch[i][j] = find_index(char)
    return index_batch, lengths

def word_2_index(batch, batch_size, dict_loc):
    w_dict = load_obj(dict_loc)
    # filter words that do not occur in the dictionary
    batch = [[word if word in w_dict else '<oov>' for word in sent] for sent in batch]
    max_sent_len = max([len(x) for x in batch])
    index_batch = np.zeros([batch_size, max_sent_len])
    lengths = []
    # load the indices for the words from the dictionary
    for i, words in enumerate(batch):
        lengths.append(len(words))
        for j, word in enumerate(words):
            index_batch[i][j] = w_dict[word]
    return index_batch, lengths

