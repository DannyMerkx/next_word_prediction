#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 12:34:49 2019
Run once to prepare the database
@author: danny
"""
from collections import defaultdict
import pickle
import os 
import csv

# location of the training database
train_loc = '/home/danny/Documents/databases/next_word_prediction/data/train.txt'
# this script saves 3 files, training data with <s> and </s> tokens, training
# data converted to embedding indices, and the dictionary mapping tokens to 
# embedding indices.
preproc_train_loc = os.path.join('/home/danny/Documents/databases/next_word_prediction/data',
                                 'train_preproc.txt')
idx_train_loc = os.path.join('/home/danny/Documents/databases/next_word_prediction/data', 
                             'train_indices.csv')
emb_dict_loc = os.path.join('/home/danny/Documents/databases/next_word_prediction/data', 
                            'train_indices')
# function to save pickled data
def save_obj(obj, loc):
    with open(f'{loc}.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL) 

# open the file with the training data
sentences = []
with open(train_loc) as file:
    for line in file:
        sentences.append(line)

# create the dictionary which will contain the embedding indices   
emb_dict = defaultdict(int)
count = 1
text_as_indices = []
for idx, sent in enumerate(sentences):
    # split the sentence and add beginning and end of sentence tokens
    words = sent.split()
    words.append('</s>')
    words.insert(0,'<s>')
    for w in words:
        if emb_dict[w] == 0:
            emb_dict[w] = count
            count += 1
    # join the sentence back together with the two new tokens
    sentences[idx] = ' '.join(words)
    # convert the text to embedding indices (potentially saves some time during 
    # network training)
    text_as_indices += [[emb_dict[x] for x in words]]
    
# save the processed text data 
with open(preproc_train_loc, mode = 'w') as file:
    for line in sentences:
        file.write(line + '\n')
        
# also save the text as converted to the indices 
with open(idx_train_loc, mode='w') as file:
    writer = csv.writer(file, delimiter=',')
    for line in text_as_indices:
        writer.writerow(line)

## save the dictionary
save_obj(emb_dict, emb_dict_loc)

