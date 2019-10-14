#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:11:23 2019

@author: mhasan8
"""


import matplotlib.pyplot as plt
import numpy as np
import os
import re
#import seaborn as sns
import pickle as pkl
import glob as glob

f="/scratch/mhasan8/processed_multimodal_data/Humor/final_humor_sdk/universal_sentence_embeddings/*.pkl"

pkl_files=glob.glob(f)

all_data={}

for file in pkl_files:    
    data=pkl.load(open(file,"rb"))    
    for hid,d in data.items():
        all_data[hid]=d
        
        
print(len(all_data))
out_file="/scratch/mhasan8/processed_multimodal_data/Humor/final_humor_sdk/universal_sentence_embedding_sdk.pkl"
pkl.dump(all_data,open(out_file,"wb"))