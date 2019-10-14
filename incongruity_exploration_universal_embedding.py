#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:25:31 2019

@author: mhasan8
"""

import pickle as pkl
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from scipy import spatial

f="/scratch/mhasan8/processed_multimodal_data/Humor/final_humor_sdk/universal_sentence_embedding_sdk.pkl"

label_f="/scratch/mhasan8/processed_multimodal_data/Humor/final_humor_sdk/humor_label_sdk.pkl"

data=pkl.load(open(f,"rb"))
labels=pkl.load(open(label_f,"rb"))

humor_dist=[]
nonhumor_dist=[]

for hid,inst in data.items():
    
    punchline_emb=inst['punchline_embedding']
    context_summary_embedding=inst["context_summary_embedding"]
    
    
    if context_summary_embedding is None:
        continue
    
    cos_dist=spatial.distance.cosine(punchline_emb,context_summary_embedding)
    
    if labels[hid]==1:
        humor_dist.append(cos_dist)
    else:
        nonhumor_dist.append(cos_dist)


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
plt.xticks([1,2],["Humor","Not Humor"])
plt.ylabel("Cosine distance between punchine & context paragraph")
plt.savefig("incongruity_use.png")
plt.show()



















