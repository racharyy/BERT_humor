#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 12:32:18 2019

@author: achattor
"""


from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import pickle
import sys
#from global_configs import *


def load_pickle(file):
    
    try:
        with open(file,'rb') as f:
            data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(file,'rb') as f:
            data = pickle.load(f,encoding='latin1')
    except Exception as e:
        print('Unable to load data',file,':',e)
        raise
    return data




data = load_pickle("/scratch/achattor/mosi/all_mod_data.pickle")
print(len(data['train']),len(data['dev']),len(data['test']))
print(len(data["train"][1][0][0]),len(data["train"][1][0][1]),len(data["train"][1][0][2]))

w,a,v = data["train"][0][0]
w1,a1,v1 = data["train"][78][0]
print(v.shape,v1.shape)