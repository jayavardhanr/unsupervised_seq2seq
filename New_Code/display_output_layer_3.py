#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 02:00:38 2018

@author: jayavardhanreddy
"""

import numpy as np
import matplotlib.pyplot as plt


# Train Data Output - Sequence Length 520
InputData = np.load(
    'outputs/Train_Test_Input_LEVEL1_ECG_PredictionTask_SeqLen_1040_hidden_20_layers_3_lr_01_batchsize_100_False.npy')
print(InputData.shape)

trueOutput = np.load(
    'outputs/Train_Test_TrueY_LEVEL1_ECG_PredictionTask_SeqLen_1040_hidden_20_layers_3_lr_01_batchsize_100_False.npy')
print(trueOutput.shape)

predictedOutput_layer_3_20 = np.load(
    'outputs/Train_Test_Prediction_LEVEL1_ECG_PredictionTask_SeqLen_1040_hidden_20_layers_3_lr_01_batchsize_100_False.npy')
print(predictedOutput_layer_3_20.shape)

# =============================================================================
# ##########    Printing True Signal and Predicted Signal in Train Set ##################
# =============================================================================
for i in range(5):
    input_data = InputData[:, i]
    print(input_data.shape)

    true = trueOutput[:, i]
    print(true.shape)

    pred_4 = predictedOutput_layer_3_20[:, i]
    print(pred_4.shape)

    plt.figure(figsize=(15, 10))
    plt.title("TestData - Input Signal")
    plt.plot(list(input_data))
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.title("TestData - True Output vs Predicted Signal")
    plt.plot(list(true))
    plt.plot(list(pred_4),label="Triple Layer 20")
    plt.legend()
    plt.show()

for i in range(15, 20):
    input_data = InputData[:, i]
    print(input_data.shape)

    true = trueOutput[:, i]
    print(true.shape)

    pred_4 = predictedOutput_layer_3_20[:, i]
    print(pred_4.shape)

    plt.figure(figsize=(15, 10))
    plt.title("TestData - Input Signal")
    plt.plot(list(input_data))
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.title("TestData - True Output vs Predicted Signal")
    plt.plot(list(true))
    plt.plot(list(pred_4),label="Triple Layer 20")
    plt.legend()
    plt.show()

############### Test Data #######################

# Train Data Output - Sequence Length 520
InputData = np.load(
    'outputs/Test_Test_Input_LEVEL1_ECG_PredictionTask_SeqLen_1040_hidden_20_layers_3_lr_01_batchsize_100_False.npy')
print(InputData.shape)

trueOutput = np.load(
    'outputs/Test_Test_TrueY_LEVEL1_ECG_PredictionTask_SeqLen_1040_hidden_20_layers_3_lr_01_batchsize_100_False.npy')
print(trueOutput.shape)

predictedOutput_layer_3_20 = np.load(
    'outputs/Test_Test_Prediction_LEVEL1_ECG_PredictionTask_SeqLen_1040_hidden_20_layers_3_lr_01_batchsize_100_False.npy')
print(predictedOutput_layer_3_20.shape)

# =============================================================================
# ##########    Printing True Signal and Predicted Signal in Test Set ##################
# =============================================================================

for i in range(5):
    input_data = InputData[:, i]
    print(input_data.shape)

    true = trueOutput[:, i]
    print(true.shape)

    pred_4 = predictedOutput_layer_3_20[:, i]
    print(pred_4.shape)

    plt.figure(figsize=(15, 10))
    plt.title("TestData - Input Signal")
    plt.plot(list(input_data))
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.title("TestData - True Output vs Predicted Signal")
    plt.plot(list(true))
    plt.plot(list(pred_4),label="Triple Layer 20")
    plt.legend()
    plt.show()

for i in range(15, 20):
    input_data = InputData[:, i]
    print(input_data.shape)

    true = trueOutput[:, i]
    print(true.shape)

    pred_4 = predictedOutput_layer_3_20[:, i]
    print(pred_4.shape)

    plt.figure(figsize=(15, 10))
    plt.title("TestData - Input Signal")
    plt.plot(list(input_data))
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.title("TestData - True Output vs Predicted Signal")
    plt.plot(list(true))
    plt.plot(list(pred_4),label="Triple Layer 20")
    plt.legend()
    plt.show()

# =============================================================================
# ####Checking Anamoly
# ##Anamoly is at Sample 50 in Test Set for Sequence Length 208
# pred = predictedOutputTest[:,50]
# print(pred.shape)
#
# true=trueOutputTest[:,50]
# print(true.shape)
#
# plt.figure()
# plt.title("TestData Checking Anamoly - True Signal")
# plt.plot(list(true))
# plt.show()
#
#
# plt.figure()
# plt.title("TestData Checking Anamoly- Predicted Signal")
# plt.plot(list(pred))
# plt.show()
# =============================================================================


