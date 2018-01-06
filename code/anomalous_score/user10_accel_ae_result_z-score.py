# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:47:22 2017

@author: Gordon
"""

from sklearn.preprocessing import normalize

import numpy as np
import matplotlib.pyplot as plt

'''
Calculate anomaly score with validation set and test set
'''

input_train_val = np.loadtxt('input_monday_trainself_nonoverlap_user10_2L_20H.txt')
#input_train_val = np.loadtxt('input_friday_nonoverlap_user10_2L_20H.txt')
output_train_val = np.loadtxt('output_monday_trainself_nonoverlap_user10_2L_20H.txt')
#output_train_val = np.loadtxt('output_friday_nonoverlap_user10_2L_20H.txt')

input_train_val_concat = []
output_train_val_concat = []

for in_val in range(len(input_train_val)):
    input_train_val_concat = np.concatenate((input_train_val_concat,input_train_val[in_val]),axis=0)
    
for out_val in range(len(output_train_val)):
    output_train_val_concat = np.concatenate((output_train_val_concat,output_train_val[out_val]),axis=0)



pred_error = (input_train_val - output_train_val)

error_sq = pred_error * pred_error
error_sq_sum = np.sum(pred_error * pred_error)

normed_error_sq = error_sq / error_sq_sum

normed_error_sq_mean = np.mean(normed_error_sq)
normed_error_sq_std = np.std(normed_error_sq)

z_score = abs((normed_error_sq - normed_error_sq_mean) / normed_error_sq_std)

z_score_concat = []

for z_val in range(len(z_score)):
    z_score_concat = np.concatenate((z_score_concat,z_score[z_val]),axis=0)

start_point = 0 # 100000, 200000
seq_length = len(z_score_concat) # 1000
#start_point = 700000 # 100000, 200000
#seq_length = 1000
mean_z_score_line = [np.mean(z_score)]*seq_length
step = np.arange(seq_length)

plt.figure(1)
plt.subplot(311)
plt.plot(input_train_val_concat[start_point + 0:start_point+seq_length])
plt.axis((0,len(input_train_val_concat[start_point + 0:start_point+seq_length]),min(input_train_val_concat),max(input_train_val_concat)))
plt.title('train - Original Input - user 10')
plt.xlabel('time step')
plt.ylabel('Accelerometer mag')

plt.subplot(312)
plt.plot(output_train_val_concat[start_point + 0:start_point+seq_length])
plt.axis((0,len(output_train_val_concat[start_point + 0:start_point+seq_length]),min(input_train_val_concat),max(input_train_val_concat)))
plt.title('train - Reconstructed Input - user 10')
plt.xlabel('time step')
plt.ylabel('Accelerometer mag')

plt.subplot(313)
plt.plot(step,z_score_concat[start_point + 0:start_point+seq_length])
plt.plot(step,mean_z_score_line,'r--')
plt.axis((0,len(z_score_concat[start_point + 0:start_point+seq_length]),0,30))
plt.title('train - Sequence of Z-score')
plt.xlabel('time step')
plt.ylabel('Z-score')
plt.subplots_adjust(top=0.99, bottom=0.01, left=0.01, right=0.99, hspace=0.8,wspace=0.5)
plt.show()

print ('duration(sec): %d sec' % (seq_length/20))
print ('duration(minute): %d minute' % (seq_length/20/60))
print ('duration(hour): %d hour' % (seq_length/20/60/60))


x = np.arange(100)
mean_z_score_line_one_seq = [np.mean(z_score)]*100
                     
n_samples = 50
np.random.seed(600)
shuffle_indices_z = np.random.permutation(np.arange(len(normed_error_sq)))
random_sample_indices_z = shuffle_indices_z[0:n_samples]

for z in range(n_samples):
    plt.figure(z)
    
    plt.subplot(311)
    plt.plot(input_train_val[random_sample_indices_z[z]])
    plt.axis((0,len(input_train_val[random_sample_indices_z[z]]),min(input_train_val_concat),20))
    plt.title('train - Original Input %d' % (random_sample_indices_z[z]))
    plt.xlabel('time step (100 samples per sequence)')
    plt.ylabel('Accelerometer magnitude')
    
    plt.subplot(312)
    plt.plot(output_train_val[random_sample_indices_z[z]])
    plt.axis((0,len(output_train_val[random_sample_indices_z[z]]),min(input_train_val_concat),20))
    plt.title('train - Reconstructed Input %d' % (random_sample_indices_z[z]))
    plt.xlabel('time step (100 samples per sequence)')
    plt.ylabel('Accelerometer magnitude')
    
    
    plt.subplot(313)
    plt.plot(x,z_score[random_sample_indices_z[z]])
    plt.plot(x,mean_z_score_line_one_seq,'r--')
    plt.axis((0,len(z_score[random_sample_indices_z[z]]),0,30))
    plt.title('train - Sequence %d' % (random_sample_indices_z[z]))
    plt.xlabel('time step (100 samples per sequence)')
    plt.ylabel('Z-score')
    
    plt.subplots_adjust(top=0.99, bottom=0.01, left=0.01, right=0.99, hspace=0.8,wspace=0.5)
    plt.show()