# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 22:33:22 2017

@author: Gordon
"""

from sklearn.preprocessing import normalize

import numpy as np
import matplotlib.pyplot as plt
import random

import scipy.io


"""
Level 1 encoder input for generating Level 2 encoder input
"""
resample_original_signals = np.load('user43_train_X_0_100.npy')

first_level_sequence_length = resample_original_signals.shape[1]
first_level_batch_size = resample_original_signals.shape[0] # number of sequence
first_level_input_output_dim = resample_original_signals.shape[2] # number of units from previous level

first_level_enc_input = np.zeros((first_level_sequence_length,first_level_batch_size,first_level_input_output_dim))

for i in range(first_level_batch_size):
    for j in range(first_level_sequence_length):
        first_level_enc_input[j][i] = resample_original_signals[i][j][0]

first_level_enc_input = first_level_enc_input[0:first_level_sequence_length, 0:first_level_batch_size-2000]

"""
LEVEL 2 input generation
"""

Lv2_input = np.load('INPUT_LEVEL2_GRU_non_overlap.npy')

Lv2_input_train = Lv2_input[:,0:24000,:]
Lv2_input_test = Lv2_input[:,24000:25800,:]

# total num sequence 26400 used
second_level_sequence_length = 60
second_level_batch_size = int(Lv2_input.shape[1] / 60)
second_level_bs_train = int(Lv2_input_train.shape[1] / 60)
second_level_bs_test = int(Lv2_input_test.shape[1] / 60)


second_level_input_output_dim = Lv2_input.shape[2]

second_level_enc_input = np.zeros((second_level_sequence_length,second_level_batch_size,second_level_input_output_dim))

second_level_enc_input_train = np.zeros((second_level_sequence_length,second_level_bs_train,second_level_input_output_dim))
second_level_enc_input_test = np.zeros((second_level_sequence_length,second_level_bs_test,second_level_input_output_dim))

start_index_each_seq = 0
for i in range(second_level_batch_size):
    for j in range(second_level_sequence_length):
        second_level_enc_input[j][i] = Lv2_input[0][start_index_each_seq+j]
    start_index_each_seq = start_index_each_seq + second_level_sequence_length

#second_level_enc_input_tr = second_level_enc_input[0:second_level_sequence_length, 0:second_level_batch_size-40]
#second_level_enc_input_test = second_level_enc_input[0:second_level_sequence_length, second_level_batch_size-40:second_level_batch_size-10]

start_index_each_seq = 0
for i in range(second_level_bs_train):
    for j in range(second_level_sequence_length):
        second_level_enc_input_train[j][i] = Lv2_input_train[0][start_index_each_seq+j]
    start_index_each_seq = start_index_each_seq + second_level_sequence_length
    
start_index_each_seq = 0
for i in range(second_level_bs_test):
    for j in range(second_level_sequence_length):
        second_level_enc_input_test[j][i] = Lv2_input_test[0][start_index_each_seq+j]
    start_index_each_seq = start_index_each_seq + second_level_sequence_length


"""
LEVEL 2 Prediction
HIDDEN_UNIT = 20
OPTIMIZER = Adam
ENCODER - non-overlapped sequences
DECODER - without input, reversing output
"""

lv2_TEST_true_reverse = np.load('TEST_LEVEL2_GRU_Adam_Y_true_reverse_1L_20u_non_overlap_without_input.npy')
lv2_TEST_true = lv2_TEST_true_reverse[::-1]

lv2_adam_50_woinput_reverse = np.load('TEST_LEVEL2_GRU_Adam_prediction_reverse_1L_50u_non_overlap_without_input.npy')
lv2_adam_50_woinput = lv2_adam_50_woinput_reverse[::-1]

lv2_adam_100_woinput_reverse = np.load('TEST_LEVEL2_GRU_Adam_prediction_reverse_1L_100u_non_overlap_without_input.npy')
lv2_adam_100_woinput = lv2_adam_100_woinput_reverse[::-1]

lv2_adam_200_woinput_reverse = np.load('TEST_LEVEL2_GRU_Adam_prediction_reverse_1L_200u_non_overlap_without_input.npy')
lv2_adam_200_woinput = lv2_adam_200_woinput_reverse[::-1]


# lv2_adam_20_woinput_reverse needs to be matched to lv1_enc_state to pass it through level 1 testing
lv2_test_enc_input = np.zeros((1,lv2_TEST_true.shape[0]*lv2_TEST_true.shape[1],lv2_TEST_true.shape[2]))
lv2_dec_output_to_lv1_test_enc_input_50 = np.zeros((1,lv2_TEST_true.shape[0]*lv2_TEST_true.shape[1],lv2_TEST_true.shape[2]))
lv2_dec_output_to_lv1_test_enc_input_100 = np.zeros((1,lv2_TEST_true.shape[0]*lv2_TEST_true.shape[1],lv2_TEST_true.shape[2]))
lv2_dec_output_to_lv1_test_enc_input_200 = np.zeros((1,lv2_TEST_true.shape[0]*lv2_TEST_true.shape[1],lv2_TEST_true.shape[2]))


lv2_TEST_Y = np.zeros((lv2_TEST_true.shape[0]*lv2_TEST_true.shape[1],lv2_TEST_true.shape[2]))
lv2_adam_prediction_50 = np.zeros((lv2_TEST_true.shape[0]*lv2_TEST_true.shape[1],lv2_TEST_true.shape[2]))
lv2_adam_prediction_100 = np.zeros((lv2_TEST_true.shape[0]*lv2_TEST_true.shape[1],lv2_TEST_true.shape[2]))
lv2_adam_prediction_200 = np.zeros((lv2_TEST_true.shape[0]*lv2_TEST_true.shape[1],lv2_TEST_true.shape[2]))
start_index = 0
for i in range(lv2_TEST_true.shape[1]):
    for j in range(lv2_TEST_true.shape[0]):
        lv2_TEST_Y[start_index+j] = lv2_TEST_true[j][i]
        lv2_adam_prediction_50[start_index+j] = lv2_adam_50_woinput[j][i]
        lv2_adam_prediction_100[start_index+j] = lv2_adam_100_woinput[j][i]
        lv2_adam_prediction_200[start_index+j] = lv2_adam_200_woinput[j][i]
        lv2_test_enc_input[0][start_index+j] = lv2_TEST_true_reverse[j][i]
        lv2_dec_output_to_lv1_test_enc_input_50[0][start_index+j] = lv2_adam_50_woinput_reverse[j][i]
        lv2_dec_output_to_lv1_test_enc_input_100[0][start_index+j] = lv2_adam_100_woinput_reverse[j][i]
        lv2_dec_output_to_lv1_test_enc_input_200[0][start_index+j] = lv2_adam_200_woinput_reverse[j][i]
    start_index = start_index + lv2_TEST_true.shape[0]

np.save('lv2_TEST_enc_input.npy',lv2_test_enc_input)
np.save('lv2_TEST_dec_output_50.npy',lv2_dec_output_to_lv1_test_enc_input_50)
np.save('lv2_TEST_dec_output_100.npy',lv2_dec_output_to_lv1_test_enc_input_100)
np.save('lv2_TEST_dec_output_200.npy',lv2_dec_output_to_lv1_test_enc_input_200)
#lv2_dec_output = np.load('lv2_TEST_dec_output.npy')

scipy.io.savemat('C:/Users/Gordon/Desktop/Unsupervised_representation_project2017/lv2_GRU_non_overlap_test_true.mat', mdict={'test_true': lv2_TEST_Y})
scipy.io.savemat('C:/Users/Gordon/Desktop/Unsupervised_representation_project2017/lv2_GRU_50u_non_overlap_prediction.mat', mdict={'test_prediction': lv2_adam_prediction_50})
scipy.io.savemat('C:/Users/Gordon/Desktop/Unsupervised_representation_project2017/lv2_GRU_100u_non_overlap_prediction.mat', mdict={'test_prediction': lv2_adam_prediction_100})
scipy.io.savemat('C:/Users/Gordon/Desktop/Unsupervised_representation_project2017/lv2_GRU_200u_non_overlap_prediction.mat', mdict={'test_prediction': lv2_adam_prediction_200})



"""
Generate Level 1 Test encoder state from Level 2 prediction
24000-25800
(1,1800,20)
"""
lv2_adam_50_woinput_reverse = np.load('TEST_LEVEL2_GRU_Adam_prediction_reverse_1L_50u_non_overlap_without_input.npy')
lv2_adam_50_woinput = lv2_adam_50_woinput_reverse[::-1]

new_enc_state = np.zeros((1,lv2_adam_50_woinput.shape[1]*lv2_adam_50_woinput.shape[0],lv2_adam_50_woinput.shape[2]))

for i in range(lv2_adam_50_woinput.shape[1]): #30
    for j in range(lv2_adam_50_woinput.shape[0]):
        new_enc_state[:,(i*lv2_adam_50_woinput.shape[0])+j] = lv2_adam_50_woinput[j][i]
                 
    



"""
Plot 1 - Level 1 test Y
Plot 2 - Level 1 test prediction
Plot 3 - Level 1 test prediction with Level 2's decoder output
"""

#TEST_new_new_LEVEL1_GRU_Adan_Y_true_reverse_1L_20u_non_overlap_TrainLoopTestLoop_without_input.npy
#TEST_LEVEL1_GRU_Adam_prediction_reverse_same_seq_1L_20u_non_overlap_TrainLoopTestLoop_without_input.npy
#TEST_new_new_LEVEL1_Cellinput_LEVEL2_Decoutput_GRU_Adam_prediction_reverse_1L_20u_non_overlap_without_input.npy

lv1_testY_reverse = np.load('TEST_new_new_LEVEL1_GRU_Adan_Y_true_reverse_1L_20u_non_overlap_TrainLoopTestLoop_without_input.npy')
lv1_testP_reverse = np.load('TEST_LEVEL1_GRU_Adam_prediction_reverse_same_seq_1L_20u_non_overlap_TrainLoopTestLoop_without_input.npy')

#lv1_testP_lv2_dec_reverse = np.load('TEST_new_50_LEVEL1_Cellinput_LEVEL2_Decoutput_GRU_Adam_prediction_reverse_1L_20u_non_overlap_without_input.npy')
#lv1_testP_lv2_dec_reverse = np.load('TEST_new_100_LEVEL1_Cellinput_LEVEL2_Decoutput_GRU_Adam_prediction_reverse_1L_20u_non_overlap_without_input.npy')
#lv1_testP_lv2_dec_reverse = np.load('TEST_new_200_LEVEL1_Cellinput_LEVEL2_Decoutput_GRU_Adam_prediction_reverse_1L_20u_non_overlap_without_input.npy')
#lv1_testP_lv2_dec_reverse = np.load('TEST_new_LEVEL1_Cellinput_LEVEL2_encinput_GRU_Adam_prediction_reverse_1L_20u_non_overlap_without_input.npy')
#lv1_testP_lv2_dec_reverse = np.load('TEST_new_LEVEL1_Cellinput_LEVEL2_predicted_encstate_GRU_Adam_prediction_reverse_1L_20u_non_overlap_without_input.npy')
lv1_testP_lv2_dec_reverse = np.load('TEST_new_LEVEL1_Cellinput_LEVEL2_predicted_encstate_GRU_Adam_prediction_reverse_1L_20u_non_overlap_without_input_30step.npy')



lv1_testY = lv1_testY_reverse[::-1]
lv1_testP = lv1_testP_reverse[::-1]
lv1_testP_lv2_dec = lv1_testP_lv2_dec_reverse[::-1]


lv1_testY_ = np.zeros((lv1_testY.shape[1],lv1_testY.shape[0]))
lv1_testP_ = np.zeros((lv1_testP.shape[1],lv1_testP.shape[0]))
lv1_testP_lv2_dec_ = np.zeros((lv1_testP_lv2_dec.shape[1],lv1_testP_lv2_dec.shape[0]))
for i in range(lv1_testY.shape[1]):
    for j in range(lv1_testY.shape[0]):
        lv1_testY_[i][j] = lv1_testY[j][i][0]
        lv1_testP_[i][j] = lv1_testP[j][i][0]
        lv1_testP_lv2_dec_[i][j] = lv1_testP_lv2_dec[j][i][0]


np.random.seed(15)
rand_nums_test = [x for x in range(lv1_testY.shape[1])]
random.shuffle(rand_nums_test)
num_display_seq = 100

seqs_examples = rand_nums_test[0:num_display_seq]
#seqs_examples = [161,171,174,544]

min_y_test = np.amin((lv1_testY_,lv1_testP_,lv1_testP_lv2_dec_))
max_y_test = np.amax((lv1_testY_,lv1_testP_,lv1_testP_lv2_dec_))

for j in range(len(seqs_examples)):
    fig = plt.figure(figsize=(6, 10))
    plt.subplot(311)
    plt.plot(lv1_testY_[seqs_examples[j]],'k')
    plt.axis((0,100,min_y_test,max_y_test))
    plt.title('Level 1 Testing - True output {}'.format(seqs_examples[j]))
    plt.subplot(312)
    plt.plot(lv1_testP_[seqs_examples[j]],'r')
    plt.axis((0,100,min_y_test,max_y_test))
    plt.title('Level 1 Testing - Prediction output {}'.format(seqs_examples[j]))
    plt.subplot(313)
    plt.plot(lv1_testP_lv2_dec_[seqs_examples[j]],'b')
    plt.axis((0,100,min_y_test,max_y_test))
    plt.title('Level 1 Testing - Prediction output with Lv2 decoder output {}'.format(seqs_examples[j]))
    
    plt.tight_layout()
    
    fig.savefig('GRU_Lv1_20_Lv2_enc_input_{}.png'.format(j))

    
#inputs_nonoverlap = np.load('user43_test_X_100_120.npy')
#full_trace = inputs_nonoverlap.reshape(len(inputs_nonoverlap),100)
#full_trace_inputs = []
#for i in range(len(full_trace)):
#    for j in range(len(full_trace[i])):
#        full_trace_inputs.append(full_trace[i][j])
#        
#full_trace_test = test_inputs.reshape(len(test_inputs),100)
#full_trace_inputs_test = []
#for i in range(len(full_trace_test)):
#    for j in range(len(full_trace_test[i])):
#        full_trace_inputs_test.append(full_trace_test[i][j])
#        
#plt.figure(figsize=(15, 10))
#plt.plot(full_trace_inputs,'b')
#plt.axis((0,len(full_trace_inputs),min(full_trace_inputs),max(full_trace_inputs)))
#plt.title('Level 1 Training set')
#
#plt.figure(figsize=(15, 10))
#plt.plot(full_trace_inputs_test,'r')
#plt.axis((0,len(full_trace_inputs_test),min(full_trace_inputs),max(full_trace_inputs)))
#plt.title('Level 1 Test set')