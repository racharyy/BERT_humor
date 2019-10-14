#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 16:14:01 2019

@author: mhasan8
"""


import pickle as pkl


dev_data=pkl.load(open("humor_bert_embeddings_dev.pkl","rb"))
train_data=pkl.load(open("humor_bert_embeddings_train.pkl","rb"))
test_data=pkl.load(open("humor_bert_embeddings_test.pkl","rb"))



humor_cls_embeddings=[]


for humor_inst in train_data:
    inst={'cls_embedding':humor_inst['cls_embedding'],'label':humor_inst['label']}
    humor_cls_embeddings.append(inst)


for humor_inst in dev_data:
    inst={'cls_embedding':humor_inst['cls_embedding'],'label':humor_inst['label']}
    humor_cls_embeddings.append(inst)


for humor_inst in test_data:
    inst={'cls_embedding':humor_inst['cls_embedding'],'label':humor_inst['label']}
    humor_cls_embeddings.append(inst)    
    
    
pkl.dump(humor_cls_embeddings,open("humor_bert_cls_embeddings.pkl","wb"))