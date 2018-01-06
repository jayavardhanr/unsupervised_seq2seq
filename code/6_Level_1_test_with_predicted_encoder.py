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

#first_level_enc_input_val = first_level_enc_input[0:first_level_sequence_length, first_level_batch_size-2000:first_level_batch_size]
#first_level_enc_input = first_level_enc_input[0:first_level_sequence_length, 0:first_level_batch_size-2000]

first_level_enc_input_test = first_level_enc_input[0:first_level_sequence_length, 24000:25800]

#lv2_TEST_dec_output = np.load('lv2_TEST_dec_output_200.npy') # (1, 1800, 20)
#lv2_TEST_dec_output = np.load('lv2_TEST_enc_input.npy') # (1, 1800, 20) checking whether new_enc_state works or not


#Lv2_input = np.load('INPUT_LEVEL2_GRU_non_overlap.npy')
#Lv2_input_test_24000_25800 = Lv2_input[:,24000:25800,:]

lv2_adam_50_woinput_reverse = np.load('TEST_LEVEL2_GRU_Adam_prediction_reverse_1L_50u_non_overlap_without_input_30step.npy')
lv2_adam_50_woinput = lv2_adam_50_woinput_reverse[::-1]

predicted_enc_state = np.zeros((1,lv2_adam_50_woinput.shape[1]*lv2_adam_50_woinput.shape[0],lv2_adam_50_woinput.shape[2]))

for i in range(lv2_adam_50_woinput.shape[1]):
    for j in range(lv2_adam_50_woinput.shape[0]):
        predicted_enc_state[:,(i*lv2_adam_50_woinput.shape[0])+j] = lv2_adam_50_woinput[j][i]


seq_length = first_level_enc_input_test.shape[0]
#batch_size = second_level_enc_input.shape[1]  # Low value used for live demo purposes - 100 and 1000 would be possible too, crank that up!

# Output dimension (e.g.: multiple signals at once, tied in time)
output_dim = input_dim = first_level_enc_input_test.shape[-1]
hidden_dim = 20  # Count of hidden neurons in the recurrent units.
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

    #new_enc_state = tf.placeholder(tf.float32, [lv2_TEST_dec_output.shape[0], lv2_TEST_dec_output.shape[1], lv2_TEST_dec_output.shape[2]])

    #new_enc_state = tf.placeholder(tf.float32, [lv2_TEST_dec_output.shape[1], lv2_TEST_dec_output.shape[2]])
    #new_enc_state = tf.placeholder(tf.float32, [Lv2_input_test_24000_25800.shape[0], Lv2_input_test_24000_25800.shape[1], Lv2_input_test_24000_25800.shape[2]])

    new_enc_state = tf.placeholder(tf.float32, [predicted_enc_state.shape[0], predicted_enc_state.shape[1], predicted_enc_state.shape[2]])


    #print "new_enc_state shape = \n",new_enc_state # 1
    #print "new_enc_state[0][0] : ", new_enc_state[0][0] # Tensor("Seq2seq/strided_slice:0", shape=(20,), dtype=float32)
    #print "new_enc_state[0][1] : ", new_enc_state[0][1] # Tensor("Seq2seq/strided_slice_1:0", shape=(20,), dtype=float32)


    #new_enc_state = tf.placeholder(tf.float32, shape=(None, lv2_TEST_dec_output.shape[0]))

    #feed_dict.update({enc_state[0][t]: lv2_TEST_dec_output[0][t] for t in range(lv2_TEST_dec_output.shape[1])})

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
    #enc_state = new_enc_state
    #new_enc_state = lv2_TEST_dec_output
    #print "encoder state shape = ",len(enc_state) # 1
    #print "enc_state[0][0] : ", enc_state[0][0] # Tensor("Seq2seq/strided_slice:0", shape=(20,), dtype=float32)
    #print "enc_state[0][1] : ", enc_state[0][1] # Tensor("Seq2seq/strided_slice_1:0", shape=(20,), dtype=float32)
    #print "new encoder state shape = ",len(new_enc_state)

    # not working because of passing entire matrix, it should pass correspodning index of 1-D array

    #dec_outputs, dec_state = tf.nn.seq2seq.rnn_decoder(dec_inp, enc_state, cell)
    #dec_outputs, dec_state = tf.nn.seq2seq.rnn_decoder(dec_inp, [new_enc_state], cell)
    dec_outputs, dec_state = tf.nn.seq2seq.rnn_decoder(dec_inp, [new_enc_state[0]], cell)


    output_scale_factor = tf.Variable(1.0, name="Output_ScaleFactor")
    # Final outputs: with linear rescaling similar to batch norm,
    # but without the "norm" part of batch normalization hehe.
    reshaped_outputs = [output_scale_factor *
                        (tf.matmul(i, w_out) + b_out) for i in dec_outputs]

saver = tf.train.Saver()
saved_path = os.path.join('checkpoints_seq2seq/', 'LEVEL1_GRU_MODEL_Adam_user43_non_overlap_1L_20u_TrainLoop_without_input_reverse')

X_test = first_level_enc_input_test
Y_test = X_test[::-1]
feed_dict = {enc_inp[t]: X_test[t] for t in range(first_level_sequence_length)}
#feed_dict.update({new_enc_state[t]: lv2_TEST_dec_output[0][t] for t in range(lv2_TEST_dec_output.shape[1])})
#feed_dict.update({enc_state[0][t]: lv2_TEST_dec_output[0][t] for t in range(100)})
#feed_dict.update({new_enc_state: lv2_TEST_dec_output[0]})
#feed_dict.update({new_enc_state: Lv2_input_test_24000_25800})
feed_dict.update({new_enc_state: predicted_enc_state})


saver.restore(sess=sess, save_path=saved_path)
pred_outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])
#np.save('TEST_LEVEL1_GRU_Adam_prediction_1L_15u_non_overlap_TrainLoopTestLoop_without_input_4100EP.npy',pred_outputs)
#np.save('TEST_new_200_LEVEL1_Cellinput_LEVEL2_Decoutput_GRU_Adam_prediction_reverse_1L_20u_non_overlap_without_input.npy',pred_outputs)
np.save('TEST_new_LEVEL1_Cellinput_LEVEL2_predicted_encstate_GRU_Adam_prediction_reverse_1L_20u_non_overlap_without_input_30step.npy',pred_outputs)
#np.save('TEST_LEVEL1_GRU_true_1L_50u_007lr_non_overlap_TrainLoopTestLoop_EP.npy',X_test)
np.save('TEST_new_LEVEL1_GRU_Adam_Y_true_reverse_1L_20u_non_overlap_TrainLoopTestLoop_without_input.npy',Y_test)
