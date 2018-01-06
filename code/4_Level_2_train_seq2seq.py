# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 12:14:31 2017

@author: Gordon
"""

import tensorflow as tf
import numpy as np

import os

# def loop(prev, _):
#     prev = tf.matmul(prev, w_out) + b_out
#     prev = tf.stop_gradient(prev)
#     return prev

"""
Lv2_input = np.load('INPUT_LEVEL2_GRU_non_overlap.npy')

# total num sequence 26400 used
second_level_sequence_length = 60
second_level_batch_size = int(Lv2_input.shape[1] / 60)
second_level_input_output_dim = Lv2_input.shape[2]

second_level_enc_input = np.zeros((second_level_sequence_length,second_level_batch_size,second_level_input_output_dim))

start_index_each_seq = 0
for i in range(second_level_batch_size):
    for j in range(second_level_sequence_length):
        second_level_enc_input[j][i] = Lv2_input[0][start_index_each_seq+j]
    start_index_each_seq = start_index_each_seq + second_level_sequence_length


second_level_enc_input_val = second_level_enc_input[0:second_level_sequence_length, second_level_batch_size-40:second_level_batch_size-10]
second_level_enc_input_tr = second_level_enc_input[0:second_level_sequence_length, 0:second_level_batch_size-40]
"""

Lv2_input = np.load('INPUT_LEVEL2_GRU_non_overlap.npy')

Lv2_input_train = Lv2_input[:,0:24000,:]
Lv2_input_test = Lv2_input[:,24000:25800,:]

# total num sequence 26400 used
second_level_sequence_length = 30
second_level_bs_train = int(Lv2_input_train.shape[1] / second_level_sequence_length)
second_level_bs_test = int(Lv2_input_test.shape[1] / second_level_sequence_length)


second_level_input_output_dim = Lv2_input.shape[2]

second_level_enc_input_train = np.zeros((second_level_sequence_length,second_level_bs_train,second_level_input_output_dim))
second_level_enc_input_test = np.zeros((second_level_sequence_length,second_level_bs_test,second_level_input_output_dim))


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




seq_length = second_level_enc_input_train.shape[0]
#batch_size = second_level_enc_input.shape[1]  # Low value used for live demo purposes - 100 and 1000 would be possible too, crank that up!

# Output dimension (e.g.: multiple signals at once, tied in time)
output_dim = input_dim = second_level_enc_input_train.shape[-1]
hidden_dim = 50  # Count of hidden neurons in the recurrent units.
# Number of stacked recurrent cells, on the neural depth axis.
layers_stacked_count = 1

# Optmizer:
learning_rate = 0.001  # Small lr helps not to diverge during training.
#learning_rate = 0.001
# How many times we perform a training step (therefore how many times we
# show a batch).


lr_decay = 0.92  # default: 0.9 . Simulated annealing.
momentum = 0.5  # default: 0.0 . Momentum technique in weights update
lambda_l2_reg = 0.003  # L2 regularization of weights - avoids overfitting


nb_iters = 50000
#nb_iters = 50000


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
    # dec_inp = [tf.zeros_like(
    #     enc_inp[0], dtype=np.float32, name="GO")] + enc_inp[:-1]

    dec_inp = [tf.zeros_like(enc_inp[0], dtype=np.float32, name="GO")] + [tf.zeros_like(enc_inp[0], dtype=np.float32) for t in range(seq_length-1)]

    #dec_inp = [tf.zeros_like(enc_inp[0], dtype=np.float32, name="GO")] + [tf.zeros_like(enc_inp[:-1], dtype=np.float32)]

    #dec_inp = [tf.zeros_like(enc_inp[0], dtype=np.float32, name="GO") for t in range(seq_length)]
    #dec_inp = [tf.zeros(tf.shape(enc_inp[0]), dtype=np.float32, name="GO") for _ in range(seq_length)]


    # Create a `layers_stacked_count` of stacked RNNs (GRU cells here).
    cells = []
    for i in range(layers_stacked_count):
        with tf.variable_scope('RNN_{}'.format(i)):
            cells.append(tf.nn.rnn_cell.GRUCell(hidden_dim))
            # cells.append(tf.nn.rnn_cell.BasicLSTMCell(...))
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
    dec_outputs, dec_state = tf.nn.seq2seq.rnn_decoder(dec_inp, enc_state, cell)


    output_scale_factor = tf.Variable(1.0, name="Output_ScaleFactor")
    # Final outputs: with linear rescaling similar to batch norm,
    # but without the "norm" part of batch normalization hehe.
    reshaped_outputs = [output_scale_factor *
                        (tf.matmul(i, w_out) + b_out) for i in dec_outputs]


with tf.variable_scope('Loss'):
    # input_ = (47378, 100, 1)
    #self.loss = tf.reduce_mean(tf.square(self.input_ - self.output_))
    loss = 0
    for _y, _Y in zip(reshaped_outputs, expected_output):
        loss += tf.reduce_mean(tf.square(_y - _Y))

with tf.variable_scope('Optimizer'):
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

"""
# Training loss and optimizer

with tf.variable_scope('Loss'):
    # L2 loss
    output_loss = 0
    for _y, _Y in zip(reshaped_outputs, expected_output):
        output_loss += tf.reduce_mean(tf.nn.l2_loss(_y - _Y))

    # L2 regularization (to avoid overfitting and to have a  better
    # generalization capacity)
    reg_loss = 0
    for tf_var in tf.trainable_variables():
        if not ("Bias" in tf_var.name or "Output_" in tf_var.name):
            reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

    loss = output_loss + lambda_l2_reg * reg_loss

with tf.variable_scope('Optimizer'):
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate, decay=lr_decay, momentum=momentum)
    train_op = optimizer.minimize(loss)
"""
saver = tf.train.Saver()
save_dir = 'checkpoints_seq2seq/'
if not os.path.exists(save_dir):
  os.makedirs(save_dir)

#save_path = os.path.join(save_dir, 'LEVEL1_GRU_MODEL_best_train_error_user43_non_overlap_2L_50u_TrainLoop')
save_path = os.path.join(save_dir, 'LEVEL2_GRU_MODEL_Adam_user43_non_overlap_1L_50u_without_input_reverse_30step')

# Training
train_losses = []
test_losses = []

sess.run(tf.global_variables_initializer())

best_val_error = 999999999999
last_improvement = 0
required_improvement = 350

batch_size = 10
num_of_batches = int(second_level_enc_input_train.shape[1]/batch_size)

for t in range(nb_iters + 1):
    train_losses_minibatch = []
    ptr = 0

    for j in range(num_of_batches):
        X = second_level_enc_input_train[0:second_level_sequence_length, ptr:ptr+batch_size]
        #Y = X
        Y = X[::-1]
        ptr+=batch_size
        feed_dict = {enc_inp[t]: X[t] for t in range(len(enc_inp))}
        feed_dict.update({expected_output[t]: Y[t] for t in range(len(expected_output))})
        _, loss_t = sess.run([train_op, loss], feed_dict)

        train_loss = loss_t
        train_losses_minibatch.append(train_loss)
        print("Step {}/{}, Mini-batch {}/{}, train loss: {}".format(t, nb_iters, j, num_of_batches, train_loss))
    each_train_loss = sum(train_losses_minibatch)
    train_losses.append(each_train_loss)

    print("Step {}/{}, sum of train loss: {}".format(t, nb_iters, each_train_loss))

    if t % 10 == 0:
        X_val = second_level_enc_input_test
        #Y_val = X_val
        Y_val = X_val[::-1]
        feed_dict = {enc_inp[t]: X_val[t] for t in range(len(enc_inp))}
        feed_dict.update({expected_output[t]: Y_val[t] for t in range(len(expected_output))})
        loss_val = sess.run([loss], feed_dict)
        #np.save('GRU_encoder_output.npy',en_out)
        #np.save('GRU_encoder_state.npy',en_state)
        val_loss = loss_val[0]
        #val_loss = loss_val
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
