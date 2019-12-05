#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:27:09 2019

This file replaces the log_frequency feature in the experiment data files with
log_frequencies counted over the training data (we have changed the database
since the initial experiments). It also fixes a small mistake in the original 
data where somehow the word Scott was replaced with Sott leading to an oov case. 

@author: danny
"""
import pandas as pd
import os
import pickle
import csv

eeg = '/home/danny/Documents/databases/next_word_prediction/data/data_EEG.csv'
et = '/home/danny/Documents/databases/next_word_prediction/data/data_ET.csv'
spr = '/home/danny/Documents/databases/next_word_prediction/data/data_SPR.csv'

freq_dict_loc = os.path.join('/home/danny/Documents/databases/next_word_prediction/data', 
                             'word_freq')

def load_obj(loc):
    with open(f'{loc}.pkl', 'rb') as f:
        obj = pickle.load(f) 
    return obj

freq_dict = load_obj(freq_dict_loc)

eeg_data = pd.read_csv(eeg, sep = '\t')
et_data = pd.read_csv(et, sep = '\t')
spr_data = pd.read_csv(spr, sep = '\t')

# for some reason the test sentence 'Scott got up and washed his plate', turned
# into 'Sott got up ...' in the data files and there is no frequency count for Sott
def rep_word(data):
    for idx, line in enumerate(data.word):
        if line == 'Sott':
            data.set_value(idx, 'word', 'Scott')

rep_word(eeg_data)
rep_word(et_data)
rep_word(spr_data)

# get now log frequencies based on the used training data. 
def get_log_freq(data, freq_dict):
    log_freq = pd.Series()
    for idx, word in enumerate(data.word):
        # ignore punctuation, these words will be rejected based on reject_word column,
        # but we might as well get the log frequency count for them
        if word[-1]== '.' or word[-1]== '?' or word[-1]== '!' or word[-1]== ',':
            word = word[:-1]
        log_freq = log_freq.append(pd.Series(freq_dict[word.lower()]), ignore_index = True)
    data.log_freq = log_freq
    
get_log_freq(eeg_data, freq_dict)
get_log_freq(et_data, freq_dict)
get_log_freq(spr_data, freq_dict)
# convert booleans to ints so that the values are compatible with julia
eeg_data.reject_data = eeg_data.reject_data * 1
eeg_data.reject_word = eeg_data.reject_word * 1
et_data.reject_data = et_data.reject_data * 1
et_data.reject_word = et_data.reject_word * 1
spr_data.reject_data = spr_data.reject_data * 1
spr_data.reject_word = spr_data.reject_word * 1

eeg_data.to_csv(f'{eeg}_new', index = False)
et_data.to_csv(f'{et}_new', index = False)
spr_data.to_csv(f'{spr}_new', index = False)
