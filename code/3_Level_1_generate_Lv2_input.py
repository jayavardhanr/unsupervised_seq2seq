import tensorflow as tf
import numpy as np

import os

# def loop(prev, _):
#     prev = tf.matmul(prev, w_out) + b_out
#     prev = tf.stop_gradient(prev)
#     return prev

resample_original_signals = np.load('user43_train_X_0_100.npy')

first_level_sequence_length = resample_original_signals.shape[1]
first_level_batch_size = resample_original_signals.shape[0] # number of sequence
first_level_input_output_dim = resample_original_signals.shape[2] # number of units from previous level

first_level_enc_input = np.zeros((first_level_sequence_length,first_level_batch_size,first_level_input_output_dim))

for i in range(first_level_batch_size):
    for j in range(first_level_sequence_length):
        first_level_enc_input[j][i] = resample_original_signals[i][j][0]

first_level_enc_input = first_level_enc_input[0:first_level_sequence_length, 0:first_level_batch_size-2000]

seq_length = first_level_enc_input.shape[0]
#batch_size = second_level_enc_input.shape[1]  # Low value used for live demo purposes - 100 and 1000 would be possible too, crank that up!

# Output dimension (e.g.: multiple signals at once, tied in time)
output_dim = input_dim = first_level_enc_input.shape[-1]
hidden_dim = 20  # Count of hidden neurons in the recurrent units.
# Number of stacked recurrent cells, on the neural depth axis.
layers_stacked_count = 1

"""
# Optmizer:
learning_rate = 0.0001  # Small lr helps not to diverge during training.
# How many times we perform a training step (therefore how many times we
# show a batch).


lr_decay = 0.92  # default: 0.9 . Simulated annealing.
momentum = 0.5  # default: 0.0 . Momentum technique in weights update
lambda_l2_reg = 0.003  # L2 regularization of weights - avoids overfitting


#nb_iters = 1
nb_iters = 10
"""

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

# Training loss and optimizer
"""
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
saved_path = os.path.join('checkpoints_seq2seq/', 'LEVEL1_GRU_MODEL_Adam_user43_non_overlap_1L_20u_TrainLoop_without_input_reverse')


X = first_level_enc_input
feed_dict = {enc_inp[t]: X[t] for t in range(len(enc_inp))}
saver.restore(sess=sess, save_path=saved_path)
en_state = np.array(sess.run([enc_state], feed_dict)[0])
np.save('LEVEL1_GRU_Adam_1L_20u_non_overlap_final_encoder_state.npy',en_state)
np.save('INPUT_LEVEL2_GRU_non_overlap.npy',en_state)

#final_en_states = np.load('LEVEL1_GRU_1L_50u_007lr_non_overlap_final_encoder_states_3500EP.npy')
#final_en_cell_state = final_en_states[0][0]
#final_en_hidden_state = final_en_states[0][1]
#np.save('INPUT_LEVEL2_GRU_007lr_non_overlap_encoder_cell_3500EP.npy',final_en_cell_state)
#np.save('INPUT_LEVEL2_GRU_007lr_non_overlap_encoder_hidden_3500EP.npy',final_en_hidden_state)
