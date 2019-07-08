#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 15:30:05 2018

@author: danny
"""
import numpy as np
from prep_text import char_2_index, word_2_index
from nltk.tokenize.nist import NISTTokenizer

# loader for the dictionary, loads a pickled dictionary.
def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)

# pastes the parsed wmt sentences back together into the original string
def char_batcher(lang_1, lang_2, batch_size, max_len = 128, shuffle = True):
    if shuffle:
        x = list(zip(lang_1, lang_2))
        np.random.shuffle(x)
        lang_1, lang_2 = zip(*x)
    for start_idx in range(0, len(lang_1) - batch_size + 1, batch_size):
        # take a batch of nodes of the given size               
        excerpt_1 = lang_1[start_idx:start_idx + batch_size]
        excerpt_2 = lang_2[start_idx:start_idx + batch_size]

        # converts the sentence to character ids. 
        l1 = char_2_index(excerpt_1, batch_size)[0]
        l2 = char_2_index(excerpt_2, batch_size)[0]
        # cut the sentence to the maximum length
        l1 = l1[:, :max_len - 2]
        l2 = l2[:, :max_len - 2]               
        # removes extra spaces and adds bos and eos tokens
        l1 = clean(l1)
        l2 = clean(l2)
        yield l1, l2

# batcher which tokenises text and converts it to given embedding indices
def token_batcher(sents, batch_size, dict_loc, max_len = 41, shuffle = True):
    if shuffle:
        np.random.shuffle(sents)
    # load a dictionary with indices
    w_dict = load_obj(dict_loc)    
    for start_idx in range(0, len(sents) - batch_size + 1, batch_size):
        # take a batch of the given batch size               
        excerpt = sents[start_idx:start_idx + batch_size]
        # tokenise the batch
        excerpt = [x.split()[:max_len] for x in excerpt]
        # converts the sentence to token ids. 
        excerpt, lengths = word_2_index(excerpt, batch_size, w_dict)      
        yield excerpt, lengths

# batcher for data that has already been converted to indices. 
def index_batcher(sents, batch_size, max_len = 41, shuffle = True):
    if shuffle:
        np.random.shuffle(sents) 
    for start_idx in range(0, len(sents) - batch_size + 1, batch_size):
        # take a batch of the given batch size               
        excerpt = sents[start_idx:start_idx + batch_size]
        # get the max sentence length for this batch
    	max_sent_len = max([len(x) for x in excerpt])
        for i, sent in enumerate(excerpt):
            # keep track of unpadded sentence length
            lengths.append(len(sent))
            # pad the sentence to the max length
            excerpt[i] = np.pad(sent, [0, max_l - len(sent)], mode = 'constant')      
        yield index_batch, lengths
        
def pad(lang):
    # pad all sents to the max lenght in the batch
    max_l = max([len(x) for x in lang])
    padded = []
    for x in lang:
        pad_len = max_l - len(x) 
        padded.append(np.pad(x,[0,pad_len], mode = 'constant'))
    return np.array(padded)

def clean(lang):
    # remove all whitespace (not in range(104,110)), remove padding(!=0),   
    # remove bos signal (for y in x[1:]) and replace with our own bos and eos ([96] + + [97])
    lang = [[y for y in x[1:] if y not in range(104, 110) and y != 0] for x in lang]
    # replace token delimiters by space(104 if y == -1 else y)
    lang = [[105] + [104 if y == -1 else y for y in x] + [106] for x in lang]
    # now repad the input
    lang = pad(lang)
    return(lang)
    
    
