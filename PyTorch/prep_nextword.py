#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 12:34:49 2019
Run once to prepare the training database
@author: danny
"""
from collections import defaultdict, Counter
import numpy as np
import pickle
import os 

# location of the training database
train_loc = '/home/danny/Documents/databases/next_word_prediction/data/train.txt'
# this script saves 3 files, training data with <s> and </s> tokens, the dictionary 
# mapping tokens to embedding indices and the word log-frequency dictionary.
preproc_train_loc = os.path.join('/home/danny/Documents/databases/next_word_prediction/data',
                                 'train_preproc.txt')

emb_dict_loc = os.path.join('/home/danny/Documents/databases/next_word_prediction/data', 
                            'train_indices')
freq_dict_loc = os.path.join('/home/danny/Documents/databases/next_word_prediction/data', 
                            'word_freq')
# function to save pickled data
def save_obj(obj, loc):
    with open(f'{loc}.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL) 

# open the file with the training data
sentences = []
with open(train_loc) as file:
    for line in file:
        sentences.append(line)

# create a frequency dictionary in order to create the log-frequency feature
# for the LMER analysis.
freq_dict = Counter(word for sent in sentences for word in sent.split())
for key in freq_dict.keys():
    freq_dict[key] = -np.log(freq_dict[key])

# create the dictionary which will contain the embedding indices   
emb_dict = defaultdict(int)
ind = 1

for idx, sent in enumerate(sentences):
    # split the sentence and add beginning and end of sentence tokens
    words = sent.split()
    words.append('</s>')
    words.insert(0,'<s>')
    for w in words:
        if emb_dict[w] == 0:
            emb_dict[w] = ind
            ind += 1  
    # join the sentence back together with the two new tokens
    sentences[idx] = ' '.join(words)

# save the processed text data 
with open(preproc_train_loc, mode = 'w') as file:
    for line in sentences:
        file.write(line + '\n')
        
save_obj(text_as_indices, idx_train_loc)

## save the index and frequency dictionary
save_obj(emb_dict, emb_dict_loc)
save_obj(freq_dict, freq_dict_loc)
