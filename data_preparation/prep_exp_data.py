#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:27:09 2019

This script add the training data logfrequency and the SUBTLEX log frequency to 
the human reading data files. It also fixes a small mistake in the original 
data where somehow the word Scott was replaced with Sott leading to an oov case. 

@author: danny
"""
import pandas as pd
import numpy as np
import os
import pickle
from collections import defaultdict
eeg = '/home/danny/Documents/databases/next_word_prediction/data/data_EEG.csv'
et = '/home/danny/Documents/databases/next_word_prediction/data/data_ET.csv'
spr = '/home/danny/Documents/databases/next_word_prediction/data/data_SPR.csv'

# spr_et are the spr data files split by whether sentences occur in the ET data
# or not
spr_et = '/home/danny/Documents/databases/next_word_prediction/data/data_SPR_ET.csv'
spr_et2 = '/home/danny/Documents/databases/next_word_prediction/data/data_SPR_ET2.csv'

freq_dict_loc = os.path.join('/home/danny/Documents/databases/next_word_prediction/data', 
                             'word_freq')

subtlex_loc = '/home/danny/Downloads/SUBTLEXusfrequencyabove1.xls'

def load_obj(loc):
    with open(f'{loc}.pkl', 'rb') as f:
        obj = pickle.load(f) 
    return obj
# load the training data frequency, this is created by prep_nextword.py
freq_dict = load_obj(freq_dict_loc)
# load subtlex word frequencies
subtlex = pd.read_excel(subtlex_loc)
subtlex_dict = {str(subtlex.iloc[idx].Word).lower(): subtlex.iloc[idx].Lg10WF for idx in range(len(subtlex))}
subtlex_dict = defaultdict(int, subtlex_dict)
# load the human reading data
eeg_data = pd.read_csv(eeg, sep = ',')
et_data = pd.read_csv(et, sep = ',')
spr_data = pd.read_csv(spr, sep = ',')

spr_et_data = pd.read_csv(spr_et, sep = ',')
spr_et2_data = pd.read_csv(spr_et2, sep = ',')

# for some reason the test sentence 'Scott got up and washed his plate', turned
# into 'Sott got up ...' in the data files and there is no frequency count for Sott
def rep_word(data):
    for idx, line in enumerate(data.word):
        if line == 'Sott':
            data.set_value(idx, 'word', 'Scott')

rep_word(eeg_data)
rep_word(et_data)
rep_word(spr_data)

rep_word(spr_et_data)
rep_word(spr_et2_data)

# get log frequencies based on the used training data. 
def get_log_freq(data, freq_dict):
    log_freq = pd.Series(dtype='float64')
    for idx, word in enumerate(data.word):
        # ignore punctuation, these words will be rejected based on reject_word column,
        # but we might as well get the log frequency count for them
        if word[-1]== '.' or word[-1]== '?' or word[-1]== '!' or word[-1]== ',':
            word = word[:-1]
        log_freq = log_freq.append(pd.Series(freq_dict[word.lower()]), ignore_index = True)
    data['log_freq'] = log_freq
# get log frequencies from subtlex
def get_subtlex_freq(data, subtlex):
    subtlex_freq = pd.Series(dtype='float64')
    for idx, word in enumerate(data.word):
        word = word.replace('\'', '').lower()
        # ignore punctuation
        if word[-1]== '.' or word[-1]== '?' or word[-1]== '!' or word[-1]== ',':
            word = word[:-1]
        #convert 10log to word counts and add 1 to each word count to prevent
        # missing words leading to log(0)
        freq = 10**subtlex[word]
        if freq != 1:
            freq += 1
        subtlex_freq = subtlex_freq.append(pd.Series(-np.log(freq)), ignore_index = True)
    data['subtlex_freq'] = subtlex_freq

get_log_freq(eeg_data, freq_dict)
get_log_freq(et_data, freq_dict)
get_log_freq(spr_data, freq_dict)

get_subtlex_freq(eeg_data, subtlex_dict)
get_subtlex_freq(et_data, subtlex_dict)
get_subtlex_freq(spr_data, subtlex_dict)

get_subtlex_freq(spr_et_data, subtlex_dict)
get_subtlex_freq(spr_et2_data, subtlex_dict)

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

spr_et_data.to_csv(f'{spr_et}_new', index = False)
spr_et2_data.to_csv(f'{spr_et2}_new', index = False)

