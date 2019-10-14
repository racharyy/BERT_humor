#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 16:17:52 2019

@author: mhasan8
"""


from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import pickle
import sys
sys.path.insert(0,'./pytorch-pretrained-BERT')
# from mosi_dataset_constants import SDK_PATH, DATA_PATH, WORD_EMB_PATH, CACHE_PATH
# import sys

# if SDK_PATH is None:
#     print("SDK path is not specified! Please specify first in constants/paths.py")
#     exit(0)
# else:
#     sys.path.append(SDK_PATH)
    
import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig,MultimodalBertForSequenceClassification, IncongruityBertForSequenceClassification,BertModel
#from pytorch_pretrained_bert.tokenization import BertTokenizer
#We are using the tokenization that amir did
from pytorch_pretrained_bert.amir_tokenization import BertTokenizer

from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_punchline , text_context=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_punchline = text_punchline
        self.text_context = text_context
        self.label = label
    def __str__(self):
        print("guid:{0},text_a:{1},text_b:{2},label:{3}".format(self.guid,self.text_punchline ,self.text_context,self.label))




# class InputExample(object):
#     """A single training/test example for simple sequence classification."""

#     def __init__(self, guid, text_a, text_b=None, label=None):
#         """Constructs a InputExample.

#         Args:
#             guid: Unique id for the example.
#             text_a: string. The untokenized text of the first sequence. For single
#             sequence tasks, only this sequence must be specified.
#             text_b: (Optional) string. The untokenized text of the second sequence.
#             Only must be specified for sequence pair tasks.
#             label: (Optional) string. The label of the example. This should be
#             specified for train and dev examples, but not for test examples.
#         """
#         self.guid = guid
#         self.text_a = text_a
#         self.text_b = text_b
#         self.label = label
#     def __str__(self):
#         print("guid:{0},text_a:{1},text_b:{2},label:{3}".format(self.guid,self.text_a,self.text_b,self.label))


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual,acoustic ,input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual=visual
        self.acoustic=acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id



def multi_collate(batch):
    '''
    Collate functions assume batch = [Dataset[i] for i in index_set]
    '''
    # for later use we sort the batch in descending order of length
    batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
    
    # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
    labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
    sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)
    visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
    acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])
    
    # lengths are useful later in using RNNs
    lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])
    return sentences, visual, acoustic, labels, lengths

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,_config):
    """Loads a data file into a list of `InputBatch`s."""
    #print("label_list:",label_list)

    label_map = {label : i for i, label in enumerate(label_list)}

    ########-----------------Not needed since we are reading the text directly----------------########

    # with open(os.path.join(_config["dataset_location"],'word2id.pickle'), 'rb') as handle:
    #     word_2_id = pickle.load(handle)
    # id_2_word = { id_:word for (word,id_) in word_2_id.items()}
    #print(id_2_word)
    

    features = []
    for (ex_index, example) in enumerate(examples):
       
        #(words, visual, acoustic), label, segment = example
        context, punchline, label = example ###-------Don't understand the segment properly
        #print("label is  -----> ", label)
        #print(words,label, segment)
        #we will look at acoustic and visual later
        #words = " ".join([id_2_word[w] for w in words])
        #print("string word:", words)
        example = InputExample(guid = None, text_punchline = punchline, text_context=context, label=label)
        #print(example)
        token_cur, inversions_cur  = tokenizer.tokenize(example.text_punchline,invertable=True)
        if len(token_cur)>max_seq_length-2:
            token_cur=token_cur[:max_seq_length-2]
        tokens = [token_cur]
        #print(len(token_cur),"+++++++++++++")
        
        #if _config['has_context']:
            
        num_context = len(example.text_context)
        if num_context<5:
            cont_list = [[] for bad in range(5-num_context)]
        else:
            cont_list = []
        for i in range(num_context):
            #print(example.text_context[i])
            contexti_token, token_inversions_cur  = tokenizer.tokenize(example.text_context[i],invertable=True)
            # Account for [CLS] and [SEP] with "- 2"
            if len(contexti_token) > max_seq_length - 2:
                contexti_token = contexti_token[:(max_seq_length - 2)]
            cont_list.append(contexti_token)            # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        #_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        tokens = cont_list+tokens
        

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        segment_ids, input_ids, input_mask = [], [], []
        num_sent = len(tokens) ### 6 when  using context 1 otherwise
        for i in range(num_sent):

            tokens[i] = ["[CLS]"] + tokens[i] + ["[SEP]"] 
            #print(tokens[i])
            #print("==============================================")
            input_id, segment_id = tokenizer.convert_tokens_to_ids(tokens[i]), [0] * len(tokens[i])
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            cur_input_mask = [1] * len(input_id)

            # Zero-pad up to the sequence length. 
            padding = [0] * (max_seq_length - len(input_id)) 
            input_id = input_id+padding
            segment_id = segment_id+padding
            cur_input_mask= cur_input_mask+padding
            #print(len(input_id),"----------------------")
            assert len(input_id) == max_seq_length
            assert len(cur_input_mask) == max_seq_length
            assert len(segment_id) == max_seq_length
            
            input_ids.append(input_id)
            segment_ids.append(segment_id) 
            input_mask.append(cur_input_mask)

            

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

       
        visual = np.ones((max_seq_length,47))
        acoustic = np.ones((max_seq_length,74))
        features.append(
                InputFeatures(input_ids=input_ids,
                              visual=visual,
                              acoustic=acoustic,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features




def get_appropriate_dataset(data,tokenizer, output_mode,_config):
    features = convert_examples_to_features(
            data, _config["label_list"],_config["max_seq_length"], tokenizer, output_mode,_config)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    
    
    all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
    all_acoustic = torch.tensor([f.acoustic for f in features], dtype=torch.float)
    
    #print("bert_ids:",all_input_ids)

    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_visual,all_acoustic,all_input_mask, all_segment_ids, all_label_ids)
    return dataset




_config={'bert_model':"bert-base-uncased", 
         'do_lower_case':True,
         'dataset_location':'/scratch/mhasan8/processed_multimodal_data/Humor',
         'max_seq_length':128,'label_list':[0,1],'output_mode':"classification",
         'device': torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu"),
         'shuffle':True,
          'num_workers':2
         }


model = BertModel.from_pretrained(_config["bert_model"])


with open(os.path.join(_config["dataset_location"],'humor_splitdata_sdk.pkl'), 'rb') as handle:
        all_data = pickle.load(handle)
        train_data = all_data["train"]
        dev_data= all_data["dev"]
        test_data=all_data["test"]
        


#humor_data=train_data+dev_data+test_data
humor_data=train_data

tokenizer = BertTokenizer.from_pretrained(_config["bert_model"], do_lower_case=_config["do_lower_case"])
output_mode = _config["output_mode"]
humor_dataset=get_appropriate_dataset(humor_data,tokenizer, output_mode,_config)

humor_dataloader = DataLoader(humor_dataset, batch_size=1,shuffle=_config["shuffle"], num_workers=_config["num_workers"])

humor_bert_embeddings=[]

model.eval()
with torch.no_grad():
        for step, batch in enumerate(tqdm(humor_dataloader, desc="Iteration")):
            batch = tuple(t.to(_config["device"]) for t in batch)
           
            input_ids, visual,acoustic,input_mask, segment_ids, label_ids = batch  
            
            encoder_output, pooled_output=model(input_ids.view(-1,_config["max_seq_length"]), segment_ids.view(-1,_config["max_seq_length"]), input_mask.view(-1,_config["max_seq_length"]), output_all_encoded_layers=False)
            
            embedding={'cls_embedding':pooled_output.cpu().numpy(),"encoder_outputs":encoder_output.cpu().numpy(),'label':label_ids.cpu().numpy()[0]}
            #embedding={'cls_embedding':pooled_output.cpu().numpy(),'label':label_ids.cpu().numpy()[0]}
            #embedding={'cls_embedding':pooled_output,"encoder_outputs":encoder_output,'label':label_ids[0]}
            humor_bert_embeddings.append(embedding)
            
            
pickle.dump(humor_bert_embeddings,open("humor_bert_embeddings_train.pkl","wb"))