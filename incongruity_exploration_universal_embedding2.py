#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 12:19:30 2019

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

d_h={0:[],1:[],2:[],3:[],4:[]}
d_nh={0:[],1:[],2:[],3:[],4:[]}

c=0
for hid,inst in data.items():
    
    context_embeddings=inst['context_embeddings']
    
    if len(context_embeddings)==0:
        continue
    
    context_embeddings=context_embeddings.tolist()
    punchline_emb=inst['punchline_embedding']
    
    
    context_embeddings.append(punchline_emb)
    
    s=6-len(context_embeddings)-1
    # print("hid:",hid)
    # print(len(context_embeddings))
    for i in range(1,len(context_embeddings)):
        prev=context_embeddings[i-1]
        curr=context_embeddings[i]
        cos_dist=spatial.distance.cosine(curr,prev)
        
        
        # print("i:",i)
        # print("index:",s+i)
            
        if labels[hid]==1.:
            d_h[s+i].append(cos_dist)
        else:
            d_nh[s+i].append(cos_dist)
        


for i in range(5):
    print(len(d_h[i]))
    print(len(d_nh[i]))
    print("*******")  


humor_distributions=[d_h[0],d_h[1],d_h[2],d_h[3],d_h[4]]

nonhumor_distributions=[d_h[0],d_nh[0],d_h[1],d_nh[1],d_h[2],d_nh[2],d_h[3],d_nh[3],d_h[4],d_nh[4]]


    
    

# print("humor:")
# print("mean",np.mean(humor_dist))
# print("std",np.std(humor_dist))

# print("non humor:")
# print("mean",np.mean(nonhumor_dist))
# print("std",np.std(nonhumor_dist))        

# data_to_plot=[humor_dist,nonhumor_dist]

fig = plt.figure(1,figsize=(9,6))

ax=fig.add_subplot(111)

bp=ax.boxplot(nonhumor_distributions)
#plt.xticks([1,2],["Humor","Not Humor"])
plt.ylabel("Cosine distance between punchine & context sentences")
plt.savefig("incongruity.png")
plt.show()



















