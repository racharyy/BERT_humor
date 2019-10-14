#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 13:02:28 2019

@author: mhasan8
"""


import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import re
#import seaborn as sns
import pickle as pkl
import sys

module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
embed = hub.Module(module_url,trainable=False)

out_file="/scratch/mhasan8/processed_multimodal_data/Humor/final_humor_sdk/universal_sentence_embeddings/"+sys.argv[1]+".pkl"
humor_sentence_embeddings_sdk={}



def get_punchline_sentence(punchline_features):
    s=""
    for w in punchline_features:
        s+=w+" "    
    return s.strip()+"."

def get_context_sentences(context_features):
    context_sentences=[]
    paragraph=" "
    for ct in context_features:
        s=""
        for w in ct:
            s+=w+" "
        s=s.strip()+"."
        context_sentences.append(s)
        paragraph+=" "+s
    
    return context_sentences,paragraph.strip()

humor_word_sdk=pkl.load(open("/scratch/mhasan8/processed_multimodal_data/Humor/final_humor_sdk/humor_word_sdk.pkl","rb"))

humor_id_list=list(humor_word_sdk.keys())

num_humor_inst=len(humor_word_sdk)
d=50
i=int(sys.argv[1])
start_i=i*d
end_i=i*d+d

if end_i>num_humor_inst:
    end_i=num_humor_inst

humor_id_list=humor_id_list[start_i:end_i]



for hid in humor_id_list:
    
    session=tf.Session() 
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    
    humor_inst=humor_word_sdk[hid]
    punchline_features=humor_inst['punchline_features']
    context_features=humor_inst['context_features']

    punchline_sentence=get_punchline_sentence(punchline_features)
    context_sentences,context_paragraph=get_context_sentences(context_features)
    
    messages=[punchline_sentence]
    punchline_embedding,context_embeddings,context_summary_embedding=None,[],None
    

    message_embeddings = session.run(embed(messages))
    punchline_embedding = np.array(message_embeddings[0])
    
    
    embeddings_inst={"punchline_embedding":punchline_embedding,"context_embeddings":context_embeddings,"context_summary_embedding":context_summary_embedding}
    
    if len(context_features)>0:

        messages=context_sentences
         
        message_embeddings = session.run(embed(messages))
        context_embeddings = np.array(message_embeddings)
        
        messages=[context_paragraph]
        message_embeddings = session.run(embed(messages))
        context_summary_embedding = np.array(message_embeddings[0])
    
        embeddings_inst['context_embeddings']=context_embeddings
        embeddings_inst['context_summary_embedding']=context_summary_embedding
        
    
    if hid%5==0:
        print("hid",hid)
    humor_sentence_embeddings_sdk[hid]=embeddings_inst
    
    session.close()

pkl.dump(humor_sentence_embeddings_sdk,open(out_file,"wb"))
