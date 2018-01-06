# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 21:18:19 2017

@author: Gordon
"""

import csv
import glob
import numpy as np
import pylab
#data_dir = "../Sample Dataset-20170228T035448Z-001/Sample Dataset/February 2016 Sample (old)/AlgoSnap Sample Dataset CSV Format/AlgoSnap Sample Dataset CSV Format/participant-one-csv/0_Accelerometer-352622063881655_1454956346105.csv"

from datetime import datetime
from math import radians, cos, sin, asin, sqrt, atan2
from itertools import groupby
from scipy import signal
import matplotlib.pyplot as plt


'''
Accelerometer (Smart phone)
'''

Accelerometer_list = glob.glob("0_Accelerometer-3*.csv")

Accelerometer_trace = []
Accel_mag_trace = []
Accel_time_trace = []
for i in range(len(Accelerometer_list)):
    with open(Accelerometer_list[i], newline='') as csvfile:
        Accelerometer_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        next(Accelerometer_reader, None)
        #next(csvfile)
        Accelerometer = []
        Accel_mag = []
        Accel_time = []
        
        for row in Accelerometer_reader:
            
            row_split = row[0].split(",")
            dt = datetime.fromtimestamp(int(row_split[2]) // 1000000000)
            s = dt.strftime('%Y-%m-%d %H:%M:%S')
            #s += '.' + str(int(1454956105656222395 % 1000000000)).zfill(9)
            row_split[2] = s
            Accelerometer.append(row_split)
            x = float(row_split[3])
            y = float(row_split[4])
            z = float(row_split[5])
            mag = sqrt(x**2 + y**2 + z**2)
            Accel_mag.append(mag)
            Accel_time.append(s)
            #print(', '.join(row))
        #del Accelerometer[0]
    Accelerometer_trace.append(Accelerometer)
    Accel_mag_trace.append(Accel_mag)
    Accel_time_trace.append(Accel_time)
    

Accel_1s_trace = []
Accel_1s_resample_trace = []
Accel_1s_downsample_trace = []
Accel_1s_not_resample_trace = []
Accel_group_trace = []
total_sec = 0
total_sample = 0
for a_t in range(len(Accel_time_trace)):
    Accel_group = []
    Accel_group = [list(j) for i, j in groupby(Accel_time_trace[a_t])]
    Accel_group_trace.append(Accel_group)
    stack_cnt = 0
    accel_1s_seq = []
    for g in range(len(Accel_group)):
        cnt = len(Accel_group[g])
        full_1s_trace =  Accel_mag_trace[a_t][stack_cnt:stack_cnt+cnt]
        resample_1s_trace = signal.resample(np.asarray(full_1s_trace[0:200]),20) # remove the values of index above 200, and resample from 200 to 20
        resample_1s_trace = resample_1s_trace -9.8
        downsample_1s_trace = signal.decimate(np.asarray(full_1s_trace[0:200]),10)
        downsample_1s_trace = downsample_1s_trace -9.8
        accel_1s_seq.append(resample_1s_trace)
        #accel_1s_seq.append(Accel_mag_trace[a_t][stack_cnt:stack_cnt+cnt])
        Accel_1s_downsample_trace.append(downsample_1s_trace)
        Accel_1s_resample_trace.append(resample_1s_trace)
        Accel_1s_not_resample_trace.append(np.asarray(full_1s_trace[0:200])-9.8)
        stack_cnt += cnt
        total_sample += cnt
    
    Accel_1s_trace.append(accel_1s_seq)
    total_sec += len(accel_1s_seq)
    
print ("Total duration(sec) = " +str(total_sec))
print ("Average number of samples per second = "+str(total_sample/total_sec))

#np.savetxt('Accel_1s_resample_trace.txt',Accel_1s_resample_trace)

np.savetxt('Accel_1s_resample_trace_subject1.txt',Accel_1s_resample_trace)

'''
Preprocess for LSTM Autoencoder
'''

#del Accel_1s_trace[0][0] # remove the first second that has less than 200 samples

#Accel_1s_resample_trace = np.loadtxt('Accel_1s_resample_trace.txt')
#Accel_1s_resample_trace = list(Accel_1s_resample_trace)

del Accel_1s_downsample_trace[0]
del Accel_1s_resample_trace[0]
del Accel_1s_not_resample_trace[0]

Accel_1s_downsample_trace_7950_sec = Accel_1s_downsample_trace[0:7950]
Accel_1s_resample_trace_7950_sec = Accel_1s_resample_trace[0:7950] # cut the last 9 seconds
Accel_1s_not_resample_trace_7950_sec = Accel_1s_not_resample_trace[0:7950]


concat_downsample_1s_trace = []
for z in range(len(Accel_1s_downsample_trace_7950_sec)):
    concat_downsample_1s_trace = np.concatenate((concat_downsample_1s_trace,Accel_1s_downsample_trace_7950_sec[z]),axis=0)

concat_resample_1s_trace = []
for j in range(len(Accel_1s_resample_trace_7950_sec)):
    concat_resample_1s_trace = np.concatenate((concat_resample_1s_trace,Accel_1s_resample_trace_7950_sec[j]),axis=0)

concat_not_resample_1s_trace = []
for w in range(len(Accel_1s_not_resample_trace_7950_sec)):
    concat_not_resample_1s_trace = np.concatenate((concat_not_resample_1s_trace,Accel_1s_not_resample_trace_7950_sec[w]),axis=0)

#def rolling_window(a, window):
#    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
#    strides = a.strides + (a.strides[-1],)
#    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
#
#def subsequences(ts, window):
#    shape = (ts.size - window + 1, window)
#    strides = ts.strides * 2
#    return np.lib.stride_tricks.as_strided(ts, shape=shape, strides=strides)    

def split_with_overlap(seq, length, overlap):
    return [seq[i:i+length] for i in range(0, len(seq), length - overlap)]

each_sequence_length = 100
overlap_length = 20
overlap_X = split_with_overlap(concat_resample_1s_trace, each_sequence_length, each_sequence_length-overlap_length)

overlap_X_new = overlap_X[0:-int(each_sequence_length/overlap_length)]
overlap_X_new_arr = np.asarray(overlap_X_new)
overlap_X_final = overlap_X_new_arr.reshape([len(overlap_X_new_arr),each_sequence_length,1])


train_X = overlap_X_final[0:len(overlap_X_final)-(round(len(overlap_X_final)/4))]
val_X = overlap_X_final[len(overlap_X_final)-(round(len(overlap_X_final)/4)):len(overlap_X_final)]

X_split = np.split(concat_resample_1s_trace,len(concat_resample_1s_trace)/100) # sequence length = 200 (10sec), 159000/200
X_split_arr = np.asarray(X_split)
X = X_split_arr.reshape([len(concat_resample_1s_trace)/100,100,1])

plt.plot(concat_not_resample_1s_trace)
plt.title('Non-Resample')
plt.xlabel('time step (200 samples per second)')
plt.ylabel('Accelerometer magnitude')
plt.show()

plt.plot(concat_resample_1s_trace)
plt.title('Resample')
plt.xlabel('time step (20 samples per second)')
plt.ylabel('Accelerometer magnitude')
plt.text(1000, 30, r'total of 7950 seconds')
plt.show()

plt.plot(concat_downsample_1s_trace,'ro')
plt.title('Downsample')
plt.xlabel('time step (20 samples per second)')
plt.ylabel('Accelerometer magnitude')
plt.text(1000, 30, r'total of 7950 seconds')
plt.show()

"""
n_samples = 20
#
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(Accel_1s_resample_trace_7950_sec)))
random_sample_indices = shuffle_indices[0:n_samples]
#
#ran_resample_trace = Accel_1s_resample_trace_7950_sec[random_sample_indices]


for iter_p in range(n_samples):
    plt.plot(Accel_1s_not_resample_trace_7950_sec[random_sample_indices[iter_p]])
    plt.title('Non-Resample %d' % (iter_p+1))
    plt.xlabel('time step (200 samples per second)')
    plt.ylabel('Accelerometer magnitude')
    plt.show()
    
    plt.plot(Accel_1s_resample_trace_7950_sec[random_sample_indices[iter_p]])
    plt.title('Resample %d' % (iter_p+1))
    plt.xlabel('time step (20 samples per second)')
    plt.ylabel('Accelerometer magnitude')
    plt.show()
    
"""

'''
Accelerometer (Smart watch)
'''
'''
Accelerometer_list = glob.glob("0_Accelerometer-c*.csv")

Accelerometer_trace = []
Accel_mag_trace = []
Accel_time_trace = []
for i in range(len(Accelerometer_list)):
    with open(Accelerometer_list[i], newline='') as csvfile:
        Accelerometer_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        next(Accelerometer_reader, None)
        #next(csvfile)
        Accelerometer = []
        Accel_mag = []
        Accel_time = []
        
        for row in Accelerometer_reader:
            
            row_split = row[0].split(",")
            dt = datetime.fromtimestamp(int(row_split[2]) // 1000000000)
            s = dt.strftime('%Y-%m-%d %H:%M:%S')
            #s += '.' + str(int(1454956105656222395 % 1000000000)).zfill(9)
            row_split[2] = s
            Accelerometer.append(row_split)
            x = float(row_split[3])
            y = float(row_split[4])
            z = float(row_split[5])
            mag = sqrt(x**2 + y**2 + z**2)
            Accel_mag.append(mag)
            Accel_time.append(s)
            #print(', '.join(row))
        #del Accelerometer[0]
    Accelerometer_trace.append(Accelerometer)
    Accel_mag_trace.append(Accel_mag)
    Accel_time_trace.append(Accel_time)
    

Accel_1s_trace = []
Accel_1s_resample_trace = []
Accel_group_trace = []
total_sec = 0
total_sample = 0
for a_t in range(len(Accel_time_trace)):
    Accel_group = []
    Accel_group = [list(j) for i, j in groupby(Accel_time_trace[a_t])]
    Accel_group_trace.append(Accel_group)
    stack_cnt = 0
    accel_1s_seq = []
    for g in range(len(Accel_group)):
        cnt = len(Accel_group[g])
        full_1s_trace =  Accel_mag_trace[a_t][stack_cnt:stack_cnt+cnt]
        resample_1s_trace = signal.resample(np.asarray(full_1s_trace[0:200]),20) # remove the values of index above 200, and resample from 200 to 20
        #downsample_1s_trace = signal.decimate(np.asarray(full_1s_trace[0:200]),10)        
        accel_1s_seq.append(resample_1s_trace)
        #accel_1s_seq.append(Accel_mag_trace[a_t][stack_cnt:stack_cnt+cnt])
        Accel_1s_resample_trace.append(resample_1s_trace)
        stack_cnt += cnt
        total_sample += cnt
    
    Accel_1s_trace.append(accel_1s_seq)
    total_sec += len(accel_1s_seq)
    
print ("Total duration(sec) = " +str(total_sec))
print ("Average number of samples per second = "+str(total_sample/total_sec))

np.savetxt('sw_Accel_1s_resample_trace.txt',Accel_1s_resample_trace)

del Accel_1s_trace[0][0] # remove the first second that has less than 200 samples
del Accel_1s_resample_trace[0]

Accel_1s_resample_trace_7270_sec = Accel_1s_resample_trace[0:7270] # cut the last 9 seconds

sw_concat_resample_1s_trace = []
for j in range(len(Accel_1s_resample_trace_7270_sec)):
    sw_concat_resample_1s_trace = np.concatenate((sw_concat_resample_1s_trace,Accel_1s_resample_trace_7270_sec[j]),axis=0)

sw_X_split = np.split(sw_concat_resample_1s_trace,len(sw_concat_resample_1s_trace)/200) # sequence length = 200 (10sec), 159000/200
sw_X_split_arr = np.asarray(sw_X_split)
sw_X = sw_X_split_arr.reshape([len(sw_concat_resample_1s_trace)/200,200,1])

plt.plot(sw_concat_resample_1s_trace)
plt.show()
'''



    #max_1s_len = max(len(l) for l in accel_1s_seq)
    #X = np.zeros((len(accel_1s_seq),max_1s_len))
#X = np.empty((len(new_accel_ex),max_1s_len),np.float32)