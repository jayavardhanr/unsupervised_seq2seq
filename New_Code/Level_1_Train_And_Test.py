#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 02:02:09 2018

@author: jayavardhanreddy
"""
import tensorflow as tf
import numpy as np
import os
import pickle
import math
import argparse

#learning_rate=0.00001,batch_size=100
def train(length,hidden_dim,layers_stacked_count,learning_rate,batch_size,flip):
    inputData = np.load('inputData/sel102_trainDataInput_'+str(length)+'_offset.npy')
    outputData = np.load('inputData/sel102_trainDataOutput_'+str(length)+'_offset.npy')
    
    print('*****************************************')
    print("Train Data Loaded")
    print('inputData',inputData.shape)
    print('outputData',outputData.shape)
    print('*****************************************')
    
    first_level_batch_size = inputData.shape[0] # number of sequences
    first_level_sequence_length = inputData.shape[1]
    first_level_input_output_dim = inputData.shape[2] # number of units from previous level
    
    first_level_enc_input = np.zeros((first_level_sequence_length,first_level_batch_size,first_level_input_output_dim))
    first_level_enc_output = np.zeros((first_level_sequence_length,first_level_batch_size,first_level_input_output_dim))
    
    for i in range(first_level_batch_size):
        for j in range(first_level_sequence_length):
            first_level_enc_input[j][i] = inputData[i][j][0]
            
    for i in range(first_level_batch_size):
        for j in range(first_level_sequence_length):
            first_level_enc_output[j][i] = outputData[i][j][0]
            
    first_level_enc_input_val = first_level_enc_input[0:first_level_sequence_length, first_level_batch_size-200:first_level_batch_size]
    first_level_enc_input = first_level_enc_input[0:first_level_sequence_length, 0:first_level_batch_size-200]
    
    first_level_enc_output_val = first_level_enc_output[0:first_level_sequence_length, first_level_batch_size-200:first_level_batch_size]
    first_level_enc_output = first_level_enc_output[0:first_level_sequence_length, 0:first_level_batch_size-200]
    
    seq_length = first_level_enc_input.shape[0]
    
    # Output dimension (e.g.: multiple signals at once, tied in time)
    output_dim = input_dim = first_level_enc_input.shape[-1]
    
    # =============================================================================
    #     lr_decay = 0.92  # default: 0.9 . Simulated annealing.
    #     momentum = 0.5  # default: 0.0 . Momentum technique in weights update
    #     lambda_l2_reg = 0.003  # L2 regularization of weights - avoids overfitting
    # =============================================================================
    
    nb_iters = 50000
    
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    
    with tf.variable_scope('Seq2seq'):
    
        # Encoder: inputs
        enc_inp = [
            tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
            for t in range(seq_length)
        ]
    
        # Decoder: expected outputs
        expected_output = [
            tf.placeholder(tf.float32, shape=(None, output_dim), name="expected_output_".format(t))
            for t in range(seq_length)
        ]
    
        # Give a "GO" token to the decoder.
        # You might want to revise what is the appended value "+ enc_inp[:-1]".
        # dec_inp = [tf.zeros_like(enc_inp[0], dtype=np.float32, name="GO")] + enc_inp[:-1]
    
        dec_inp = [tf.zeros_like(enc_inp[0], dtype=np.float32, name="GO")] + [tf.zeros_like(enc_inp[0], dtype=np.float32) for t in range(seq_length-1)]
    
        # Create a `layers_stacked_count` of stacked RNNs (GRU cells here).
        cells = []
        for i in range(layers_stacked_count):
            with tf.variable_scope('RNN_{}'.format(i)):
                cells.append(tf.nn.rnn_cell.GRUCell(hidden_dim))
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    
        # For reshaping the input and output dimensions of the seq2seq RNN:
        w_in = tf.Variable(tf.random_normal([input_dim, hidden_dim]))
        b_in = tf.Variable(tf.random_normal([hidden_dim], mean=1.0))
        w_out = tf.Variable(tf.random_normal([hidden_dim, output_dim]))
        b_out = tf.Variable(tf.random_normal([output_dim]))
        
        
        '''
        Jay- We don't use this anywhere
        '''
        reshaped_inputs = [tf.nn.relu(tf.matmul(i, w_in) + b_in) for i in enc_inp]
    
        # Here, the encoder and the decoder uses the same cell, HOWEVER,
        # the weights aren't shared among the encoder and decoder, we have two
        # sets of weights created under the hood according to that function's def.
        enc_outputs, enc_state = tf.nn.rnn(cell, enc_inp, dtype=tf.float32)
        # =============================================================================
        #         print('\n\n*******************************\n\n')
        #         print('\n\n*******************************\n\n')
        #         print('\n\n*******************************\n\n')
        #         print('\n\n*******************************\n\n')
        #         print(enc_outputs)
        #         print('\n\n*******************************\n\n')
        #         print('\n\n*******************************\n\n')
        #         print('\n\n*******************************\n\n')
        #            
        #         print('\n\n*******************************\n\n')
        #         print(enc_state)
        #         print('\n\n*******************************\n\n')
        #         print('\n\n*******************************\n\n')
        #         print('\n\n*******************************\n\n')
        #         print('\n\n*******************************\n\n')
        # =============================================================================
        if flip=='yes':
            enc_state_new=()
            for i in range(layers_stacked_count):
                enc_state_new=(enc_state[i],)+enc_state_new
        else:
            enc_state_new=enc_state

            
        dec_outputs, dec_state = tf.nn.seq2seq.rnn_decoder(dec_inp, enc_state_new, cell)
        # loop_function: If not None, this function will be applied to the i-th output in order to generate the i+1-st input, 
        #and decoder_inputs will be ignored, except for the first element ("GO" symbol)
    
        #dec_outputs, dec_state = tf.nn.seq2seq.rnn_decoder(dec_inp, enc_state, cell, loop_function=loop)
    
    
        output_scale_factor = tf.Variable(1.0, name="Output_ScaleFactor")
        # Final outputs: with linear rescaling similar to batch norm,
        # but without the "norm" part of batch normalization hehe.
        '''
        Jay- we don't use reshaped inputs, but we measure error using  reshaped outputs
        '''
        reshaped_outputs = [output_scale_factor *
                            (tf.matmul(i, w_out) + b_out) for i in dec_outputs]
    
    
    with tf.variable_scope('Loss'):
        loss = 0
        #length_of_sequence=len(reshaped_outputs)
        for _y, _Y in zip(reshaped_outputs, expected_output):
            loss += tf.reduce_mean(tf.square(_y - _Y))
            
        #if length_of_sequence!=0:
        #    loss/=length_of_sequence
        
    
    with tf.variable_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss)
    
    saver = tf.train.Saver()
    save_dir = 'checkpoints_seq2seq/'
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    
    checkpoints_path='LEVEL1_ECG_PredictionTask_SeqLen_'+str(length)+'_hidden_'+str(hidden_dim)+'_layers_'+str(layers_stacked_count)+'_lr_01_batchsize_'+str(batch_size)+'_'+str(flip)
    print('Checkpoints Path for Training: ',checkpoints_path)
    save_path = os.path.join(save_dir,checkpoints_path)
    
    print('*****************************************')
    print("Training Started")
    print('*****************************************')
    
    # Training
    train_losses = []
    test_losses = []
    
    sess.run(tf.global_variables_initializer())
    
    best_val_error = float('inf')
    last_improvement = 0
    required_improvement = 100
    
    nan_flag=False
    num_of_batches = int(first_level_enc_input.shape[1]/batch_size)
    for t in range(nb_iters + 1):
        train_losses_minibatch = []
        ptr = 0
    
        for j in range(num_of_batches):
            X = first_level_enc_input[0:first_level_sequence_length, ptr:ptr+batch_size]
            Y = first_level_enc_output[0:first_level_sequence_length, ptr:ptr+batch_size]
            ptr+=batch_size
            feed_dict = {enc_inp[t]: X[t] for t in range(len(enc_inp))}
            feed_dict.update({expected_output[t]: Y[t] for t in range(len(expected_output))})
            _, loss_t = sess.run([train_op, loss], feed_dict)
            
            if math.isnan(loss_t):
                nan_flag=True
                print("********** NAN Loss ******")
                print(feed_dict)
                print(train_losses_minibatch)
                print("Step {}/{}, Mini-batch {}/{}".format(t, nb_iters, j, num_of_batches))
                break
            else:
                print(loss_t,"Is not NAN")
            
            train_loss = loss_t
            train_losses_minibatch.append(train_loss)
            print("Step {}/{}, Mini-batch {}/{}, train loss: {}".format(t, nb_iters, j, num_of_batches, train_loss))
        
        if nan_flag:
            break
        
        each_train_loss = sum(train_losses_minibatch)/len(train_losses_minibatch)
        train_losses.append(each_train_loss)
        print("Step {}/{}, sum of train loss: {}".format(t, nb_iters, each_train_loss))
    
        if t % 10 == 0:
            X_val = first_level_enc_input_val
            Y_val = first_level_enc_output_val
            feed_dict = {enc_inp[t]: X_val[t] for t in range(len(enc_inp))}
            feed_dict.update({expected_output[t]: Y_val[t] for t in range(len(expected_output))})
            loss_val = sess.run([loss], feed_dict)
    
            val_loss = loss_val[0]
            test_losses.append(val_loss)
            print("Step {}/{}, train loss: {}, \tvalidation loss: {}".format(t, nb_iters, each_train_loss, val_loss))
    
            if val_loss < best_val_error:
                saver.save(sess=sess, save_path=save_path)
                best_val_error = val_loss
                last_improvement = t
                improved_str = '*****'
            else:
                improved_str = '------'
            print "Step {}{}".format(t, improved_str)
        if t - last_improvement > required_improvement:
            print "No improvement found in a while, stopping optimization."
            print("Step {}/{}, train loss: {}, \tBEST validation loss: {}".format(t, nb_iters, each_train_loss, best_val_error))
            break
        
    print('*****************************************')
    print("Saving Losses",length,layers_stacked_count,hidden_dim,flip)
    print('*****************************************\n')
    
    save_dir_losses = 'losses/'
    if not os.path.exists(save_dir_losses):
      os.makedirs(save_dir_losses)
      
    with open('losses/'+checkpoints_path+'_trainloss', 'wb') as fp:
        pickle.dump(train_losses, fp)
        
    with open('losses/'+checkpoints_path+'_validationloss', 'wb') as fp:
        pickle.dump(test_losses, fp)
        
    print('*****************************************')
    print("Saved Losses",length,layers_stacked_count,hidden_dim,flip)
    print('*****************************************\n')
    
    
    print('*****************************************')
    print("Training Complete",length,layers_stacked_count,hidden_dim,flip)
    print('*****************************************')
        
        
def test(length,hidden_dim,layers_stacked_count,learning_rate,dataset,batch_size,flip):
    
    inputData = np.load('inputData/sel102_'+dataset+'DataInput_'+str(length)+'_offset.npy')
    outputData = np.load('inputData/sel102_'+dataset+'DataOutput_'+str(length)+'_offset.npy')
    
    first_level_batch_size = inputData.shape[0] # number of sequences
    first_level_sequence_length = inputData.shape[1]
    first_level_input_output_dim = inputData.shape[2] # number of units from previous level
    
    first_level_enc_input = np.zeros((first_level_sequence_length,first_level_batch_size,first_level_input_output_dim))
    first_level_enc_output = np.zeros((first_level_sequence_length,first_level_batch_size,first_level_input_output_dim))
    
    for i in range(first_level_batch_size):
        for j in range(first_level_sequence_length):
            first_level_enc_input[j][i] = inputData[i][j][0]
            
    for i in range(first_level_batch_size):
        for j in range(first_level_sequence_length):
            first_level_enc_output[j][i] = outputData[i][j][0]
    
    first_level_enc_input_test = first_level_enc_input
    first_level_enc_output_test = first_level_enc_output
    
    seq_length = first_level_enc_input_test.shape[0]
    
    # Output dimension (e.g.: multiple signals at once, tied in time)
    output_dim = input_dim = first_level_enc_input_test.shape[-1]
    
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    
    with tf.variable_scope('Seq2seq'):
    
        # Encoder: inputs
        enc_inp = [
            tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
            for t in range(seq_length)
        ]
    
        # Give a "GO" token to the decoder.
        # You might want to revise what is the appended value "+ enc_inp[:-1]".
        dec_inp = [tf.zeros_like(enc_inp[0], dtype=np.float32, name="GO")] + [tf.zeros_like(enc_inp[0], dtype=np.float32) for t in range(seq_length-1)]
        # dec_inp = [tf.zeros_like(
        #     enc_inp[0], dtype=np.float32, name="GO")] + enc_inp[:-1]
    
        # Create a `layers_stacked_count` of stacked RNNs (GRU cells here).
        cells = []
        for i in range(layers_stacked_count):
            with tf.variable_scope('RNN_{}'.format(i)):
                cells.append(tf.nn.rnn_cell.GRUCell(hidden_dim))
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    
        # For reshaping the input and output dimensions of the seq2seq RNN:
        w_in = tf.Variable(tf.random_normal([input_dim, hidden_dim]))
        b_in = tf.Variable(tf.random_normal([hidden_dim], mean=1.0))
        w_out = tf.Variable(tf.random_normal([hidden_dim, output_dim]))
        b_out = tf.Variable(tf.random_normal([output_dim]))
    
        reshaped_inputs = [tf.nn.relu(tf.matmul(i, w_in) + b_in) for i in enc_inp]
    
        # Here, the encoder and the decoder uses the same cell, HOWEVER,
        # the weights aren't shared among the encoder and decoder, we have two
        # sets of weights created under the hood according to that function's def.
        enc_outputs, enc_state = tf.nn.rnn(cell, enc_inp, dtype=tf.float32)
        
        if flip=='yes':
            enc_state_new=()
            for i in range(layers_stacked_count):
                enc_state_new=(enc_state[i],)+enc_state_new
        else:
            enc_state_new=enc_state

        dec_outputs, dec_state = tf.nn.seq2seq.rnn_decoder(dec_inp, enc_state_new, cell)
    
    
        output_scale_factor = tf.Variable(1.0, name="Output_ScaleFactor")
        # Final outputs: with linear rescaling similar to batch norm,
        # but without the "norm" part of batch normalization hehe.
        reshaped_outputs = [output_scale_factor *
                            (tf.matmul(i, w_out) + b_out) for i in dec_outputs]
    
    saver = tf.train.Saver()
    checkpoints_path='LEVEL1_ECG_PredictionTask_SeqLen_'+str(length)+'_hidden_'+str(hidden_dim)+'_layers_'+str(layers_stacked_count)+'_lr_01_batchsize_'+str(batch_size)+'_'+str(flip)
    saved_path = os.path.join('checkpoints_seq2seq/',checkpoints_path)
    
    X_test = first_level_enc_input_test
    Y_test = first_level_enc_output_test
    feed_dict = {enc_inp[t]: X_test[t] for t in range(first_level_sequence_length)}
    saver.restore(sess=sess, save_path=saved_path)
    en_state, pred_outputs = sess.run([enc_state, reshaped_outputs], feed_dict)
    
    if dataset=="train":
        print('*****************************************')
        print("Saving Train Outputs",length,layers_stacked_count,hidden_dim,flip)
        print('*****************************************')
        
        np.save('outputs/Train_Test_Input_'+checkpoints_path+'.npy',X_test)
        np.save('outputs/Train_Test_Encoder_State_'+checkpoints_path+'.npy',en_state)
        np.save('outputs/Train_Test_Prediction_'+checkpoints_path+'.npy',pred_outputs)
        np.save('outputs/Train_Test_TrueY_'+checkpoints_path+'.npy',Y_test)
        
        print('*****************************************')
        print("Train Testing Complete",length,layers_stacked_count,hidden_dim,flip)
        print('*****************************************')
    
    else:
        print('*****************************************')
        print("Saving Test Outputs",length,layers_stacked_count,hidden_dim,flip)
        print('*****************************************')
        
        np.save('outputs/Test_Test_Input_'+checkpoints_path+'.npy',X_test)
        np.save('outputs/Test_Test_Encoder_State_'+checkpoints_path+'.npy',en_state)
        np.save('outputs/Test_Test_Prediction_'+checkpoints_path+'.npy',pred_outputs)
        np.save('outputs/Test_Test_TrueY_'+checkpoints_path+'.npy',Y_test)
        
        print('*****************************************')
        print("Test Testing Complete",length,layers_stacked_count,hidden_dim,flip)
        print('*****************************************')
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collecting Parameters")
    
    # input parsing parameters
    parser.add_argument('-length', '--LENGTH', type=int, default=1024, help='length of sequence')
    parser.add_argument('-hidden', '--HIDDEN_DIM', type=int, default=20, help='hidden dimension')
    parser.add_argument('-layers', '--LAYERS_STACKED_COUNT', type=int, default=1, help='Numebr of Stacked layers')
    parser.add_argument('-flip', '--FLIP', type=str, default='No', help='flip')
    parser.add_argument('-batch', '--BATCH_SIZE', type=int, default=100, help='batch size')
    parser.add_argument('-lr', '--LEARNING_RATE', type=float, default=0.00001, help='learning rate')
    
    args = parser.parse_args()
    print('args:',args)
    length=args.LENGTH
    hidden_dim=args.HIDDEN_DIM
    layers_stacked_count=args.LAYERS_STACKED_COUNT
    flip=args.FLIP
    batch_size=args.BATCH_SIZE
    lr=args.LEARNING_RATE
    
    print("Training: ",hidden_dim," ",layers_stacked_count," ",flip)
    train(length,hidden_dim,layers_stacked_count,lr,batch_size,flip)
    
    print("Testing Train Data: ",hidden_dim," ",layers_stacked_count," ",flip)
    test(length,hidden_dim,layers_stacked_count,lr,'train',batch_size,flip)
    
    print("Testing Test Data: ",hidden_dim," ",layers_stacked_count," ",flip)
    test(length,hidden_dim,layers_stacked_count,lr,'test',batch_size,flip)
