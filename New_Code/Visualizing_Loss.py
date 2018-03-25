#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 12:56:15 2018

@author: jayavardhanreddy
"""

import pickle
import matplotlib.pyplot as plt


filepath1='losses/LEVEL1_ECG_PredictionTask_SeqLen_520_hidden_20_layers_3_lr_01_batchsize_1000_False_trainloss'
filepath2='losses/LEVEL1_ECG_PredictionTask_SeqLen_520_hidden_20_layers_3_lr_01_batchsize_1000_False_validationloss'


with open(filepath1, 'rb') as fp:
    a=pickle.load(fp)
    
with open(filepath2, 'rb') as fp:
    b=pickle.load(fp)
    
    
plt.figure(figsize=(10,10))
plt.plot(list(a),label=" Training Loss")

plt.legend()
plt.show()


plt.figure(figsize=(10,10))
plt.plot(list(b),label=" Validation Loss")
plt.legend()
plt.show()
        
