# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 15:13:30 2017

@author: Gordon
"""
import glob
import numpy as np
#data_dir = "../Sample Dataset-20170228T035448Z-001/Sample Dataset/February 2016 Sample (old)/AlgoSnap Sample Dataset CSV Format/AlgoSnap Sample Dataset CSV Format/participant-one-csv/0_Accelerometer-352622063881655_1454956346105.csv"

from datetime import datetime
from math import radians, cos, sin, asin, sqrt, atan2
from itertools import groupby
from scipy import signal
import matplotlib.pyplot as plt
import json


'''
Accelerometer (Smart phone) User 10
'''

accelerometer_list = glob.glob("c-accelerometer*.json")

accel_time_trace = []

accel_time_one_sample_trace = []

accel_group_trace = []

accel_mag_trace = []

accel_time_trace_datetime = []

accel_time_group_by_sec_trace = []

accel_mag_group_by_sec_trace = []

for i in range(len(accelerometer_list)):
#for i in range(10):
    lines = [line for line in open(accelerometer_list[i])]
    js = [json.loads(line) for line in lines]
    
    accel_time = []
    accel_time_one_sample = []
    accel_mag = []
    for j in range(len(js)):
        
        accel_time_convert = []
        for z in range(len(js[j]['timestamp'])):
            if len(str(js[j]['timestamp'][z])) < 19:
                real_time = 1000000 * js[j]['timestamp'][z]
            else:
                real_time = js[j]['timestamp'][z]
            dt = datetime.fromtimestamp(real_time // 1000000000)
            accel_time_convert.append(dt.strftime('%Y-%m-%d %H:%M:%S'))
            accel_time_trace_datetime.append(dt)
        
        accel_time_one_sample.append(accel_time_convert[0][0:16])
        accel_time.append(accel_time_convert)
        accel_mag.append(np.sqrt(np.array(js[j]['x'])*np.array(js[j]['x']) + np.array(js[j]['y'])*np.array(js[j]['y']) + np.array(js[j]['z'])*np.array(js[j]['z'])))
    
    accel_group = []
    accel_group = [list(j) for i, j in groupby(accel_time_one_sample)]
    accel_time_group_by_sec = []
    accel_mag_group_by_sec = []
    start_ind = 0
    end_ind = len(accel_group[0])
    for w in range(len(accel_group)):
        try:
            accel_time_group_by_sec.append(accel_time[start_ind:end_ind])
            accel_mag_group_by_sec.append(accel_mag[start_ind:end_ind])
            start_ind = end_ind
            end_ind = end_ind + len(accel_group[w+1])
        except IndexError:
            break

    accel_time_group_by_sec_trace.append(accel_time_group_by_sec)
    accel_mag_group_by_sec_trace.append(accel_mag_group_by_sec)
    accel_time_trace.append(accel_time)
    accel_time_one_sample_trace.append(accel_time_one_sample)
    accel_group_trace.append(accel_group)
    accel_mag_trace.append(accel_mag)


accel_group_by_min_resample_trace = []
accel_group_by_min_resample_trace_time = []
#final_resample_1s_overlap_trace = []
#concat_each_sec_overlap_resample_new = []

for z in range(len(accel_mag_group_by_sec_trace)):
    for z_sub in range(len(accel_mag_group_by_sec_trace[z])):
        accel_group_by_min = []
        accel_group_by_min_time = []
        for z_sub_sub in range(len(accel_mag_group_by_sec_trace[z][z_sub])):
            full_1s_trace = accel_mag_group_by_sec_trace[z][z_sub][z_sub_sub]
            if all(i <=50 for i in full_1s_trace):
                if len(full_1s_trace) < 2:
                    resample_1s_trace = np.asarray([full_1s_trace[0]]*20)
                    resample_1s_trace = resample_1s_trace - 9.8
                    accel_group_by_min.append(resample_1s_trace)
                    accel_group_by_min_time.append(accel_time_group_by_sec_trace[z][z_sub][z_sub_sub])
                else:
                    resample_1s_trace = signal.resample(np.asarray(full_1s_trace),20)
                    resample_1s_trace = resample_1s_trace - 9.8
                    accel_group_by_min.append(resample_1s_trace)
                    accel_group_by_min_time.append(accel_time_group_by_sec_trace[z][z_sub][z_sub_sub])
#        concat_resample_1s_trace = []
#        for q in range(len(accel_group_by_min)):
#            concat_resample_1s_trace = np.concatenate((concat_resample_1s_trace,accel_group_by_min[q]),axis=0)
#        each_sec_overlap_resample = split_with_overlap(concat_resample_1s_trace,each_sequence_length,each_sequence_length-overlap_length)
#        each_sec_overlap_resample_new = each_sec_overlap_resample[0:-int((each_sequence_length/overlap_length)-1)]
                                                                  
        accel_group_by_min_resample_trace.append(accel_group_by_min)
        accel_group_by_min_resample_trace_time.append(accel_group_by_min_time)
#        final_resample_1s_overlap_trace.append(each_sec_overlap_resample_new)
#        concat_each_sec_overlap_resample_new.extend(each_sec_overlap_resample_new)        
## Check accel_group_by_sec_resample_trace_time & final_resample_1s_overlap_trace to split day by day (count length of list in final_resample_1s_overlap_trace)
## concat_each_sec_overlap_resample_new[0:1000] 1000 from sum up length of list to the end of the day
#concat_each_sec_overlap_resample_new_day = np.asarray(concat_each_sec_overlap_resample_new[0:1000])
#concat_each_sec_overlap_resample_new_day_test = np.asarray(concat_each_sec_overlap_resample_new[1000:2000])
#X = concat_each_sec_overlap_resample_new_day.reshape([len(concat_each_sec_overlap_resample_new_day),each_sequence_length,1])

np.save('user10_accel_group_by_min_resample_trace.npy',accel_group_by_min_resample_trace)
np.save('user10_accel_group_by_min_resample_trace_time.npy',accel_group_by_min_resample_trace_time)




accel_group_by_min_resample_trace = np.load('user10_accel_group_by_min_resample_trace.npy')
accel_group_by_min_resample_trace_time = np.load('user10_accel_group_by_min_resample_trace_time.npy')



#accel_group_by_min_resample_trace_time[0][0][0] 00:21 min [1][0][0] 00:23 min [2][0][0] 00:27 min [100][0][0] 02:56 min
# [0:1624] 1st day(9/22) [1627:3214] 2nd day(9/23) [3215:4400] 3rd day(9/24) [4401:5729] 4th day(9/25) [5730:7479] 5th day(9/26) [7480:9485] 6th day(9/27) [9486:11101] 7th day(9/28) [11102:11947] 8th day(9/29) [11948:13167] 9th day(9/30) [13168:13177] 10th day(10/1)

'''
Before generate the dataset, plot the candidate day's trace to check patterns are useful or not.
'''
_X_train_ = accel_group_by_min_resample_trace[11948:13167]
train_concat_resample_1min_trace = []
for x_tr in range(len(_X_train_)):
    for x_tr_sub in range(len(_X_train_[x_tr])):
        train_concat_resample_1min_trace = np.concatenate((train_concat_resample_1min_trace,_X_train_[x_tr][x_tr_sub]),axis=0)
        
plt.plot(train_concat_resample_1min_trace)
plt.axis((0,len(train_concat_resample_1min_trace),-10,30))
plt.title('Resampled accel sequence (training set)')
plt.xlabel('time step (20 samples per second)')
plt.ylabel('Accelerometer magnitude')
plt.show()

_X_test_ = accel_group_by_min_resample_trace[355:500]
test_concat_resample_1min_trace = []
for x_tr in range(len(_X_test_)):
    for x_tr_sub in range(len(_X_test_[x_tr])):
        test_concat_resample_1min_trace = np.concatenate((test_concat_resample_1min_trace,_X_test_[x_tr][x_tr_sub]),axis=0)
        
plt.plot(test_concat_resample_1min_trace)
plt.axis((0,len(test_concat_resample_1min_trace),-10,30))
plt.title('Resampled accel sequence (test set)')
plt.xlabel('time step (20 samples per second)')
plt.ylabel('Accelerometer magnitude')
plt.show()


"""
'''
Generate training set and test set (training set: overlapping, test set: non-overlapping)
'''
def split_with_overlap(seq, length, overlap):
    return [seq[i:i+length] for i in range(0, len(seq), length - overlap)]

each_sequence_length = 100
overlap_length = 20

extend_each_min_overlap_X_train = []
X_train_ = accel_group_by_min_resample_trace[0:355]
#X_train_ = accel_group_by_sec_resample_trace[0:1000]
for x_tr in range(len(X_train_)):
    concat_resample_1min_trace = []
    for x_tr_sub in range(len(X_train_[x_tr])):
        concat_resample_1min_trace = np.concatenate((concat_resample_1min_trace,X_train_[x_tr][x_tr_sub]),axis=0)
    each_min_overlap_resample = split_with_overlap(concat_resample_1min_trace,each_sequence_length,each_sequence_length-overlap_length)
    each_min_overlap_resample_new = each_min_overlap_resample[0:-int((each_sequence_length/overlap_length)-1)]
    extend_each_min_overlap_X_train.extend(each_min_overlap_resample_new)                         

extend_each_min_overlap_X_train_arr = np.asarray(extend_each_min_overlap_X_train)
X_train = extend_each_min_overlap_X_train_arr.reshape([len(extend_each_min_overlap_X_train_arr),each_sequence_length,1])

    

extend_each_min_X_test = []
X_test_ = accel_group_by_min_resample_trace[355:500]
#X_test_ = accel_group_by_sec_resample_trace[355:500]
for x_t in range(len(X_test_)):
    concat_resample_1min_trace = []
    for x_t_sub in range(len(X_test_[x_t])):
        concat_resample_1min_trace = np.concatenate((concat_resample_1min_trace,X_test_[x_t][x_t_sub]),axis=0)
    last_cut = len(concat_resample_1min_trace)-(len(concat_resample_1min_trace) % each_sequence_length)
    if last_cut > 0:
        concat_resample_1min_trace_last_cut = concat_resample_1min_trace[0:last_cut]
        X_split = np.split(concat_resample_1min_trace_last_cut,int(len(concat_resample_1min_trace_last_cut)/each_sequence_length))
        extend_each_min_X_test.extend(X_split)
    else:
        pass
    
extend_each_min_X_test_arr = np.asarray(extend_each_min_X_test)
X_test = extend_each_min_X_test_arr.reshape([len(extend_each_min_X_test_arr),each_sequence_length,1])

"""



#train_accel_time_trace = accel_time_trace[33:77] # 2nd day Friday

#test_accel_time_trace = accel_time_trace[78:108] # 3rd day Saturday
#val_accel_time_trace = accel_time_trace[53:70] # 4th day
#test_accel_time_trace = accel_time_trace[109:145] # 4th day Sunday

#test_accel_time_trace = accel_time_trace[146:183] # 5th day Monday



#train_accel_mag_trace = accel_mag_trace[33:77]
#test_accel_mag_trace = accel_mag_trace[146:183]
#val_accel_mag_trace = accel_mag_trace[53:70]

#np.save('train_accel_time_trace.npy',train_accel_time_trace)
#np.save('test_accel_time_trace.npy',test_accel_time_trace)
#np.save('train_accel_mag_trace.npy',train_accel_mag_trace)
#np.save('test_accel_mag_trace.npy',test_accel_mag_trace)

#np.save('user10_accel_time_trace_datetime.npy',accel_time_trace_datetime)
"""
def split_with_dataset(trace, split_size):
    
    accel_1s_original_trace = []
    accel_1s_resample_trace = []
    
    for a_t in range(len(trace)):
        for a_t_sub in range(len(trace[a_t])):
            full_1s_trace = trace[a_t][a_t_sub]
            if len(full_1s_trace) < 2:
                resample_1s_trace = np.asarray([full_1s_trace[0]]*20)
                resample_1s_trace = resample_1s_trace - 9.8
                accel_1s_original_trace.append(full_1s_trace)
                accel_1s_resample_trace.append(resample_1s_trace)
                
            else:
                accel_1s_original_trace.append(full_1s_trace)
                resample_1s_trace = signal.resample(np.asarray(full_1s_trace),20)
                resample_1s_trace = resample_1s_trace - 9.8
                accel_1s_resample_trace.append(resample_1s_trace)
            
            
    last_cut = len(accel_1s_resample_trace)-(len(accel_1s_resample_trace) % 10)
    accel_1s_resample_trace_last_cut = accel_1s_resample_trace[0:last_cut]

    concat_resample_1s_trace = []
    for j in range(len(accel_1s_resample_trace_last_cut)):
        concat_resample_1s_trace = np.concatenate((concat_resample_1s_trace,accel_1s_resample_trace_last_cut[j]),axis=0)
    
    X_split = np.split(concat_resample_1s_trace,len(concat_resample_1s_trace)/split_size) # sequence length = 200 (10sec), 159000/200
    X_split_arr = np.asarray(X_split)
    X = X_split_arr.reshape([len(concat_resample_1s_trace)/split_size,split_size,1])
    
    return X, concat_resample_1s_trace, accel_1s_original_trace, accel_1s_resample_trace
    
train_X, concat_train_X, original_trace_train_X, resample_trace_train_X = split_with_dataset(train_accel_mag_trace,100)
test_X, concat_test_X, original_trace_test_X, resample_trace_test_X = split_with_dataset(test_accel_mag_trace,100)
#val_X, concat_val_X, original_trace_val_X, resample_trace_val_X = split_with_dataset(val_accel_mag_trace,100)



def split_with_overlap(seq, length, overlap):
    return [seq[i:i+length] for i in range(0, len(seq), length - overlap)]

each_sequence_length = 100
overlap_length = 20
train_X_overlap = split_with_overlap(concat_train_X, each_sequence_length, each_sequence_length-overlap_length)

train_X_overlap_new = train_X_overlap[0:-int(each_sequence_length/overlap_length)]
train_X_overlap_new_arr = np.asarray(train_X_overlap_new)
train_X_overlap_final = train_X_overlap_new_arr.reshape([len(train_X_overlap_new_arr),each_sequence_length,1])


test_X_overlap = split_with_overlap(concat_test_X, each_sequence_length, each_sequence_length-overlap_length)

test_X_overlap_new = test_X_overlap[0:-int(each_sequence_length/overlap_length)]
test_X_overlap_new_arr = np.asarray(test_X_overlap_new)
test_X_overlap_final = test_X_overlap_new_arr.reshape([len(test_X_overlap_new_arr),each_sequence_length,1])
np.save('user10_train_X_overlap.npy',test_X_overlap_final)

np.save('user10_test_X.npy',train_X)
np.save('user10_train_X.npy',test_X)

np.save('user10_test_X_overlap.npy',train_X_overlap_final)

plt.plot(concat_train_X)
plt.title('Resampled accel sequence (test set)')
plt.xlabel('time step (20 samples per second)')
plt.ylabel('Accelerometer magnitude')
#plt.text(1000, 30, r'total of 7950 seconds')
plt.show()

plt.plot(concat_test_X)
plt.title('Resampled accel sequence (training set)')
plt.xlabel('time step (20 samples per second)')
plt.ylabel('Accelerometer magnitude')
#plt.text(1000, 30, r'total of 7950 seconds')
plt.show()

#plt.plot(concat_val_X)
#plt.title('Resampled accel sequence (val set)')
#plt.xlabel('time step (20 samples per second)')
#plt.ylabel('Accelerometer magnitude')
##plt.text(1000, 30, r'total of 7950 seconds')
#plt.show()


save_index = []
for r in range(len(train_X)):
    for r_sub in range(len(train_X[r])):
        if train_X[r][r_sub][0] > 50:
            save_index.append((r,r_sub))

"""