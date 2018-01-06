import tensorflow as tf
import numpy as np

import os

# def loop(prev, _):
#     prev = tf.matmul(prev, w_out) + b_out
#     prev = tf.stop_gradient(prev)
#     return prev

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


seq_length = second_level_enc_input_test.shape[0]
#batch_size = second_level_enc_input.shape[1]  # Low value used for live demo purposes - 100 and 1000 would be possible too, crank that up!

# Output dimension (e.g.: multiple signals at once, tied in time)
output_dim = input_dim = second_level_enc_input_test.shape[-1]
hidden_dim = 50  # Count of hidden neurons in the recurrent units.
# Number of stacked recurrent cells, on the neural depth axis.
layers_stacked_count = 1

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
    # instead of passing enc_state, passing 2nd level decoder output (predicted output)
    dec_outputs, dec_state = tf.nn.seq2seq.rnn_decoder(dec_inp, enc_state, cell)


    output_scale_factor = tf.Variable(1.0, name="Output_ScaleFactor")
    # Final outputs: with linear rescaling similar to batch norm,
    # but without the "norm" part of batch normalization hehe.
    reshaped_outputs = [output_scale_factor *
                        (tf.matmul(i, w_out) + b_out) for i in dec_outputs]

saver = tf.train.Saver()
saved_path = os.path.join('checkpoints_seq2seq/', 'LEVEL2_GRU_MODEL_Adam_user43_non_overlap_1L_50u_without_input_reverse_30step')

X_test = second_level_enc_input_test
#Y_test = X_test[::-1]
feed_dict = {enc_inp[t]: X_test[t] for t in range(second_level_sequence_length)}
saver.restore(sess=sess, save_path=saved_path)
pred_outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])
#np.save('TEST_LEVEL1_GRU_Adam_prediction_1L_15u_non_overlap_TrainLoopTestLoop_without_input_4100EP.npy',pred_outputs)
np.save('TEST_LEVEL2_GRU_Adam_prediction_reverse_1L_50u_non_overlap_without_input_30step.npy',pred_outputs)
#np.save('TEST_LEVEL1_GRU_true_1L_50u_007lr_non_overlap_TrainLoopTestLoop_EP.npy',X_test)
#np.save('TEST_LEVEL2_GRU_Adam_Y_true_reverse_1L_50u_non_overlap_without_input.npy',Y_test)
