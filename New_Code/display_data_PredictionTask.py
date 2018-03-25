#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 02:00:38 2018

@author: jayavardhanreddy
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# ###################    Loading Train Data   ############### 
# =============================================================================
# =============================================================================
# 
# #Train Data Output - Sequence Length 208, hidden = 20
# InputData=np.load('outputs/Train_Test_Input_LEVEL1_ECG_PredictionTask_SeqLength_208_hidden_20_offset.npy')
# print(InputData.shape)
# 
# trueOutput=np.load('outputs/Train_Test_TrueY_LEVEL1_ECG_PredictionTask_SeqLength_208_hidden_20_offset.npy')
# print(trueOutput.shape)
# 
# predictedOutput=np.load('outputs/Train_Test_Prediction_LEVEL1_ECG_PredictionTask_SeqLength_208_hidden_20_offset.npy')
# print(predictedOutput.shape)
# 
# =============================================================================

# =============================================================================
# #Train Data Output - Sequence Length 312, hidden = 20
# InputData=np.load('outputs/Train_Test_Input_LEVEL1_ECG_PredictionTask_SeqLength_312_hidden_20_offset.npy')
# print(InputData.shape)
# 
# trueOutput=np.load('outputs/Train_Test_TrueY_LEVEL1_ECG_PredictionTask_SeqLength_312_hidden_20_offset.npy')
# print(trueOutput.shape)
# 
# predictedOutput=np.load('outputs/Train_Test_Prediction_LEVEL1_ECG_PredictionTask_SeqLength_312_hidden_20_offset.npy')
# print(predictedOutput.shape)
# =============================================================================

# =============================================================================
# #Train Data Output - Sequence Length 416, hidden = 25
# InputData=np.load('outputs/Train_Test_Input_LEVEL1_ECG_PredictionTask_SeqLength_416_hidden_25_offset.npy')
# print(InputData.shape)
# 
# trueOutput=np.load('outputs/Train_Test_TrueY_LEVEL1_ECG_PredictionTask_SeqLength_416_hidden_25_offset.npy')
# print(trueOutput.shape)
# 
# predictedOutput=np.load('outputs/Train_Test_Prediction_LEVEL1_ECG_PredictionTask_SeqLength_416_hidden_25_offset.npy')
# print(predictedOutput.shape)
# =============================================================================

# =============================================================================
# #Train Data Output - Sequence Length 520, hidden = 30
# InputData=np.load('outputs/Train_Test_Input_LEVEL1_ECG_PredictionTask_SeqLength_520_hidden_30_offset.npy')
# print(InputData.shape)
# 
# trueOutput=np.load('outputs/Train_Test_TrueY_LEVEL1_ECG_PredictionTask_SeqLength_520_hidden_30_offset.npy')
# print(trueOutput.shape)
# 
# predictedOutput=np.load('outputs/Train_Test_Prediction_LEVEL1_ECG_PredictionTask_SeqLength_520_hidden_30_offset.npy')
# print(predictedOutput.shape)
# 
# #Train Data Output - Sequence Length 520, hidden = 15, layers=2
# InputData_layer_2=np.load('outputs/Train_Test_Input_LEVEL1_ECG_PredictionTask_SeqLen_520_hidden_15_layers_2_lr_01_batchsize_1000.npy')
# print(InputData_layer_2.shape)
# 
# trueOutput_layer_2=np.load('outputs/Train_Test_TrueY_LEVEL1_ECG_PredictionTask_SeqLen_520_hidden_15_layers_2_lr_01_batchsize_1000.npy')
# print(trueOutput_layer_2.shape)
# 
# predictedOutput_layer_2=np.load('outputs/Train_Test_Prediction_LEVEL1_ECG_PredictionTask_SeqLen_520_hidden_15_layers_2_lr_01_batchsize_1000.npy')
# print(predictedOutput_layer_2.shape)
# 
# #Train Data Output - Sequence Length 520, hidden = 10, layers=3
# InputData_layer_3=np.load('outputs/Train_Test_Input_LEVEL1_ECG_PredictionTask_SeqLen_520_hidden_10_layers_3_lr_01_batchsize_1000.npy')
# print(InputData_layer_3.shape)
# 
# trueOutput_layer_3=np.load('outputs/Train_Test_TrueY_LEVEL1_ECG_PredictionTask_SeqLen_520_hidden_10_layers_3_lr_01_batchsize_1000.npy')
# print(trueOutput_layer_3.shape)
# 
# predictedOutput_layer_3=np.load('outputs/Train_Test_Prediction_LEVEL1_ECG_PredictionTask_SeqLen_520_hidden_10_layers_3_lr_01_batchsize_1000.npy')
# print(predictedOutput_layer_3.shape)
# =============================================================================


#Train Data Output - Sequence Length 1040
InputData=np.load('outputs/Train_Test_Input_LEVEL1_ECG_PredictionTask_SeqLen_1040_hidden_80_layers_1_lr_01_batchsize_1000.npy')
print(InputData.shape)

trueOutput=np.load('outputs/Train_Test_TrueY_LEVEL1_ECG_PredictionTask_SeqLen_1040_hidden_80_layers_1_lr_01_batchsize_1000.npy')
print(trueOutput.shape)

predictedOutput_layer_1_80=np.load('outputs/Train_Test_Prediction_LEVEL1_ECG_PredictionTask_SeqLen_1040_hidden_80_layers_1_lr_01_batchsize_1000.npy')
print(predictedOutput_layer_1_80.shape)

predictedOutput_layer_1_60=np.load('outputs/Train_Test_Prediction_LEVEL1_ECG_PredictionTask_SeqLen_1040_hidden_60_layers_1_lr_01_batchsize_1000.npy')
print(predictedOutput_layer_1_60.shape)

predictedOutput_layer_2_30=np.load('outputs/Train_Test_Prediction_LEVEL1_ECG_PredictionTask_SeqLen_1040_hidden_30_layers_2_lr_01_batchsize_1000.npy')
print(predictedOutput_layer_2_30.shape)

predictedOutput_layer_3_20=np.load('outputs/Train_Test_Prediction_LEVEL1_ECG_PredictionTask_SeqLen_1040_hidden_20_layers_3_lr_01_batchsize_1000.npy')
print(predictedOutput_layer_3_20.shape)

# =============================================================================
# ##########    Printing True Signal and Predicted Signal in Train Set ##################
# =============================================================================
for i in range(5):
    input_data=InputData[:,i]
    print(input_data.shape)
    
    true=trueOutput[:,i]
    print(true.shape)
   
    pred_1 = predictedOutput_layer_1_80[:,i]
    print(pred_1.shape)
    
    pred_2 = predictedOutput_layer_1_60[:,i]
    print(pred_2.shape)
    
    pred_3 = predictedOutput_layer_2_30[:,i]
    print(pred_3.shape)
    
    pred_4=predictedOutput_layer_3_20[:,i]
    print(pred_4.shape)
    
    plt.figure(figsize=(15,10))
    plt.title("TrainData - Input Signal")
    plt.plot(list(input_data))
    plt.show()
    
    plt.figure(figsize=(15,10))
    plt.title("TrainData - True Output vs Predicted Signal")
    plt.plot(list(true))
    #plt.plot(list(pred_1),label="Single Layer 80")
    #plt.plot(list(pred_2),label="Single Layer 60")
    #plt.plot(list(pred_3),label="Double Layer 30")
    plt.plot(list(pred_4),label="Triple Layer 20")
    plt.legend()
    plt.show()
    
for i in range(15,20):
    input_data=InputData[:,i]
    print(input_data.shape)
    
    true=trueOutput[:,i]
    print(true.shape)
   
    pred_1 = predictedOutput_layer_1_80[:,i]
    print(pred_1.shape)
    
    pred_2 = predictedOutput_layer_1_60[:,i]
    print(pred_2.shape)
    
    pred_3 = predictedOutput_layer_2_30[:,i]
    print(pred_3.shape)
    
    pred_4=predictedOutput_layer_3_20[:,i]
    print(pred_4.shape)
    
    plt.figure(figsize=(15,10))
    plt.title("TestData - Input Signal")
    plt.plot(list(input_data))
    plt.show()
    
    plt.figure(figsize=(15,10))
    plt.title("TrainData - True Output vs Predicted Signal")
    plt.plot(list(true))
    #plt.plot(list(pred_1),label="Single Layer 80")
    #plt.plot(list(pred_2),label="Single Layer 60")
    #plt.plot(list(pred_3),label="Double Layer 30")
    plt.plot(list(pred_4),label="Triple Layer 20")
    plt.legend()
    plt.show()
    
    
# =============================================================================
# ###################    Loading Test Data   ############### 
# =============================================================================

# =============================================================================
# #Test Data Output - Sequence Length 208, hidden = 20
# InputData=np.load('outputs/Test_Test_Input_LEVEL1_ECG_PredictionTask_SeqLength_208_hidden_20_offset.npy')
# print(InputData.shape)
# 
# trueOutput=np.load('outputs/Test_Test_TrueY_LEVEL1_ECG_PredictionTask_SeqLength_208_hidden_20_offset.npy')
# print(trueOutput.shape)
# 
# predictedOutput=np.load('outputs/Test_Test_Prediction_LEVEL1_ECG_PredictionTask_SeqLength_208_hidden_20_offset.npy')
# print(predictedOutput.shape)
# =============================================================================
    
    
# =============================================================================
# #Test Data Output - Sequence Length 312, hidden = 20
# InputData=np.load('outputs/Test_Test_Input_LEVEL1_ECG_PredictionTask_SeqLength_312_hidden_20_offset.npy')
# print(InputData.shape)
# 
# trueOutput=np.load('outputs/Test_Test_TrueY_LEVEL1_ECG_PredictionTask_SeqLength_312_hidden_20_offset.npy')
# print(trueOutput.shape)
# 
# predictedOutput=np.load('outputs/Test_Test_Prediction_LEVEL1_ECG_PredictionTask_SeqLength_312_hidden_20_offset.npy')
# print(predictedOutput.shape)
# =============================================================================

# =============================================================================
# #Test Data Output - Sequence Length 416, hidden = 25
# InputData=np.load('outputs/Test_Test_Input_LEVEL1_ECG_PredictionTask_SeqLength_416_hidden_25_offset.npy')
# print(InputData.shape)
# 
# trueOutput=np.load('outputs/Test_Test_TrueY_LEVEL1_ECG_PredictionTask_SeqLength_416_hidden_25_offset.npy')
# print(trueOutput.shape)
# 
# predictedOutput=np.load('outputs/Test_Test_Prediction_LEVEL1_ECG_PredictionTask_SeqLength_416_hidden_25_offset.npy')
# print(predictedOutput.shape)
# =============================================================================

#Test Data Output - Sequence Length 520, hidden = 30
InputData=np.load('outputs/Test_Test_Input_LEVEL1_ECG_PredictionTask_SeqLen_1040_hidden_80_layers_1_lr_01_batchsize_1000.npy')
print(InputData.shape)

trueOutput=np.load('outputs/Test_Test_TrueY_LEVEL1_ECG_PredictionTask_SeqLen_1040_hidden_80_layers_1_lr_01_batchsize_1000.npy')
print(trueOutput.shape)

predictedOutput_layer_1_80=np.load('outputs/Test_Test_Prediction_LEVEL1_ECG_PredictionTask_SeqLen_1040_hidden_80_layers_1_lr_01_batchsize_1000.npy')
print(predictedOutput_layer_1_80.shape)

predictedOutput_layer_1_60=np.load('outputs/Test_Test_Prediction_LEVEL1_ECG_PredictionTask_SeqLen_1040_hidden_60_layers_1_lr_01_batchsize_1000.npy')
print(predictedOutput_layer_1_60.shape)

predictedOutput_layer_2_30=np.load('outputs/Test_Test_Prediction_LEVEL1_ECG_PredictionTask_SeqLen_1040_hidden_30_layers_2_lr_01_batchsize_1000.npy')
print(predictedOutput_layer_2_30.shape)

predictedOutput_layer_3_20=np.load('outputs/Test_Test_Prediction_LEVEL1_ECG_PredictionTask_SeqLen_1040_hidden_20_layers_3_lr_01_batchsize_1000.npy')
print(predictedOutput_layer_3_20.shape)


# =============================================================================
# ##########    Printing True Signal and Predicted Signal in Test Set ##################
# =============================================================================

for i in range(5):
    input_data=InputData[:,i]
    print(input_data.shape)
    
    true=trueOutput[:,i]
    print(true.shape)
   
    pred_1 = predictedOutput_layer_1_80[:,i]
    print(pred_1.shape)
    
    pred_2 = predictedOutput_layer_1_60[:,i]
    print(pred_2.shape)
    
    pred_3 = predictedOutput_layer_2_30[:,i]
    print(pred_3.shape)
    
    pred_4=predictedOutput_layer_3_20[:,i]
    print(pred_4.shape)
    
    plt.figure(figsize=(15,10))
    plt.title("TestData - Input Signal")
    plt.plot(list(input_data))
    plt.show()
    
    plt.figure(figsize=(15,10))
    plt.title("TestData - True Output vs Predicted Signal")
    plt.plot(list(true))
    #plt.plot(list(pred_1),label="Single Layer 80")
    #plt.plot(list(pred_2),label="Single Layer 60")
    #plt.plot(list(pred_3),label="Double Layer 30")
    plt.plot(list(pred_4),label="Triple Layer 20")
    plt.legend()
    plt.show()
    
for i in range(15,20):
    input_data=InputData[:,i]
    print(input_data.shape)
    
    true=trueOutput[:,i]
    print(true.shape)
   
    pred_1 = predictedOutput_layer_1_80[:,i]
    print(pred_1.shape)
    
    pred_2 = predictedOutput_layer_1_60[:,i]
    print(pred_2.shape)
    
    pred_3 = predictedOutput_layer_2_30[:,i]
    print(pred_3.shape)
    
    pred_4=predictedOutput_layer_3_20[:,i]
    print(pred_4.shape)
    
    plt.figure(figsize=(15,10))
    plt.title("TestData - Input Signal")
    plt.plot(list(input_data))
    plt.show()
    
    plt.figure(figsize=(15,10))
    plt.title("TestData - True Output vs Predicted Signal")
    plt.plot(list(true))
    #plt.plot(list(pred_1),label="Single Layer 80")
    #plt.plot(list(pred_2),label="Single Layer 60")
    #plt.plot(list(pred_3),label="Double Layer 30")
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


