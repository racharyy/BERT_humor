#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:00:39 2019

@author: mhasan8
"""


import pickle as pkl
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import math

word_embedding_indexes_sdk=pkl.load(open("/scratch/mhasan8/processed_multimodal_data/Humor/final_humor_sdk/word_embedding_indexes_sdk.pkl","rb"))

word_embedding_list=pkl.load(open("/scratch/mhasan8/processed_multimodal_data/Humor/final_humor_sdk/word_embedding_list.pkl","rb"))

humor_label_sdk=pkl.load(open("/scratch/mhasan8/processed_multimodal_data/Humor/final_humor_sdk/humor_label_sdk.pkl","rb"))

humor_word_sdk=pkl.load(open("/scratch/mhasan8/processed_multimodal_data/Humor/final_humor_sdk/humor_word_sdk.pkl","rb"))


humor_dist=[]
nonhumor_dist=[]

for hid,data in word_embedding_indexes_sdk.items():
    
    punchline_embedding_indexes=data['punchline_embedding_indexes']    
    context_embedding_indexes=data['context_embedding_indexes']
    
    if len(context_embedding_indexes)==0:
        continue
    
    punchline_embeddings= np.array([ word_embedding_list[widx] for widx in punchline_embedding_indexes])
        
    context_embeddings=[] 
    
    for ct in context_embedding_indexes:
        ct_embeddings=np.array([word_embedding_list[widx] for widx in ct])
        context_embeddings.append(ct_embeddings)
    
    
    cos_dist_list=[]
    
    for pwe in punchline_embeddings:
         for ct in context_embeddings:
             for cwe in ct:
                 cos_dist=spatial.distance.cosine(pwe,cwe)
                 if not math.isnan(cos_dist):
                     cos_dist_list.append(cos_dist)
     
    if len(cos_dist_list)==0:
        continue
    
    if humor_label_sdk[hid]==1:
        humor_dist.append(max(cos_dist_list))
    else:
        nonhumor_dist.append(max(cos_dist_list))
        
    
    if hid%5==0:
        print(hid)



print("humor:")
print("mean",np.mean(humor_dist))
print("std",np.std(humor_dist))

print("non humor:")
print("mean",np.mean(nonhumor_dist))
print("std",np.std(nonhumor_dist))        

data_to_plot=[humor_dist,nonhumor_dist]

fig = plt.figure(1,figsize=(9,6))

ax=fig.add_subplot(111)

bp=ax.boxplot(data_to_plot)

plt.show()