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

# prepare the dictionary and data for the next word prediction task

# open the file with the training data
text = []
with open('/home/danny/Downloads/train.txt') as file:
    for line in file:
        text.append(line)

# create the dictionary which will contain the embedding indices   
def_dict = defaultdict(int)
lens = []
count = 1
for x in range(len(text)):
    # split the sentence and add beginning and ending of sentence tokens
    sent = text[x].split()
    sent.append('</s>')
    sent.insert(0,'<s>')
    lens.append(len(sent))
    for y in sent:
        if def_dict[y] == 0:
            def_dict[y] = count
            count += 1
    # join the sentence back together with the two new tokens
    text[x] = ' '.join(sent)
    
# save the processed data
with open('/home/danny/Downloads/train_preproc.txt', 'w') as file:
    for line in text:
        file.write(line + '\n')


def save_obj(obj, loc):
    with open(loc + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL) 
## save the dictionary
save_obj(def_dict, os.path.join('/home/danny/Downloads', 'train_indices'))

lens.sort()
# print the lenght of the max len sentence. Setting this correctly during training
# can save on computation time. 
print(lens[-1])
