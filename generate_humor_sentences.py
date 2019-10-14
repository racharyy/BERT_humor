#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:03:38 2019

@author: mhasan8
"""


import matplotlib.pyplot as plt
import numpy as np
import os
import re
#import seaborn as sns
import pickle as pkl
import sys

humor_word_sdk=pkl.load(open("/scratch/mhasan8/processed_multimodal_data/Humor/final_humor_sdk/humor_word_sdk.pkl","rb"))
#covarep_features_sdk=pkl.load(open("/scratch/mhasan8/processed_multimodal_data/Humor/final_humor_sdk/covarep_features_sdk.pkl","rb"))

def get_punchline_sentence(punchline_features):
    s=""
    for w in punchline_features:
        s+=w+" "    
    return s.strip()+" ."

def get_context_sentences(context_features):
    context_sentences=[]
    paragraph=" "
    for ct in context_features:
        s=""
        for w in ct:
            s+=w+" "
        s=s.strip()+" ."
        context_sentences.append(s)
        paragraph+=" "+s
    
    return context_sentences,paragraph.strip()


for hid in humor_word_sdk.keys():

    humor_inst=humor_word_sdk[hid]

    punchline_features=humor_inst['punchline_features']
    context_features=humor_inst['context_features']

    punchline_sentence=get_punchline_sentence(punchline_features)
    context_sentences,context_paragraph=get_context_sentences(context_features)
    
    
    humor_word_sdk[hid]['punchline_sentence']=punchline_sentence 
    humor_word_sdk[hid]['context_sentences']=context_sentences
    humor_word_sdk[hid]['context_paragraph']=context_paragraph 
    


out_file="/scratch/mhasan8/processed_multimodal_data/Humor/final_humor_sdk/humor_word_sentence_sdk.pkl"

#pkl.dump(humor_word_sdk,open(out_file,"wb"))