# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 20:19:25 2017

@author: Gordon
"""

import csv
import glob
import numpy as np
from numpy import ndarray
import pylab
#data_dir = "../Sample Dataset-20170228T035448Z-001/Sample Dataset/February 2016 Sample (old)/AlgoSnap Sample Dataset CSV Format/AlgoSnap Sample Dataset CSV Format/participant-one-csv/0_Accelerometer-352622063881655_1454956346105.csv"

from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt, atan2
from itertools import groupby
from scipy import signal
import matplotlib.pyplot as plt

'''
Interval Label
'''
Interval_label_list = glob.glob("0_IntervalLabel-3*.csv")
Interval_label_trace = []
Interval_time_trace = []

for i in range(len(Interval_label_list)):
    with open(Interval_label_list[i], newline='') as csvfile:
        Interval_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(Interval_reader, None)
        Interval_time = []
        Interval_label = []
        for row in Interval_reader:
            #row_split = row[0].split(",")
            start_dt = datetime.fromtimestamp(int(float(row[3])) // 1000000000)
            end_dt = datetime.fromtimestamp(int(float(row[5])) // 1000000000)
            start_s = start_dt.strftime('%Y-%m-%d %H:%M:%S')
            end_s = end_dt.strftime('%Y-%m-%d %H:%M:%S')
            Interval_label.append(row[2])
            Interval_time.append((start_dt,end_dt))
    Interval_label_trace.append(Interval_label)
    Interval_time_trace.append(Interval_time)

# end time: datetime.datetime(2016, 2, 8, 15, 25, 43) start time: datetime.datetime(2016, 2, 8, 13, 28, 52)
diff_interval = Interval_time_trace[-1][0][1] - Interval_time_trace[0][0][0]
known_interval = divmod(diff_interval.days * 86400 + diff_interval.seconds, 60)
total_sec_known_interval = (known_interval[0]*60) + known_interval[1]
                            
'''
Accelerometer (Smart phone)
'''

Accelerometer_list = glob.glob("0_Accelerometer-3*.csv")

Accelerometer_trace = []
Accel_mag_trace = []
Accel_time_trace = []
Accel_time_trace_datetime = []
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
            Accel_time_trace_datetime.append(dt)
            #print(', '.join(row))
        #del Accelerometer[0]
    Accelerometer_trace.append(Accelerometer)
    Accel_mag_trace.append(Accel_mag)
    Accel_time_trace.append(Accel_time)

    
diff_accel_time = Accel_time_trace_datetime[-1] - (Accel_time_trace_datetime[0]+timedelta(0,1))
total_interval = divmod(diff_accel_time.days * 86400 + diff_accel_time.seconds, 60)
last_cut = ((total_interval[0]*60) + total_interval[1])-(((total_interval[0]*60) + total_interval[1]) % 10)
#total_sec_interval = (total_interval[0]*60) + total_interval[1] - 9  # remove the first second and last 9 seconds
total_sec_interval = last_cut

label_time_traces = np.ndarray((total_sec_interval,20),int)
label_time_traces = label_time_traces.astype('str')
                  
#label_time_traces = np.zeros((total_sec_interval,20), dtype=np.str)
starting_time = Accel_time_trace_datetime[0] + timedelta(0,1)

seq_time_slot = []
seq_time_slot.append(starting_time.strftime('%Y-%m-%d %H:%M:%S'))
curr_time = starting_time

for i in range(total_sec_interval-1):
    next_time = curr_time + timedelta(0,1)
    seq_time_slot.append(next_time.strftime('%Y-%m-%d %H:%M:%S'))
    curr_time = next_time

    
#new_Interval_time_trace = Interval_time_trace[0:11] # for test
#new_Interval_label_trace = Interval_label_trace[0:11] # for test
new_Interval_time_trace = []
new_Interval_label_trace = []
for t in range(len(Interval_time_trace)):
    for t_i in range(len(Interval_time_trace[t])):
        new_Interval_time_trace.append(Interval_time_trace[t][t_i])
        new_Interval_label_trace.append(Interval_label_trace[t][t_i])

for j in range(len(new_Interval_time_trace)):
    start_index = seq_time_slot.index(new_Interval_time_trace[j][0].strftime('%Y-%m-%d %H:%M:%S'))
    end_index = seq_time_slot.index(new_Interval_time_trace[j][1].strftime('%Y-%m-%d %H:%M:%S'))
    duration = end_index - start_index
    for w in range(duration+1):
        label_time_traces[start_index+w,:] = new_Interval_label_trace[j]


unique_label = list(set(new_Interval_label_trace))
# Unknown = 9, Washing Hands = 1, Sitting = 2, Running = 3, Eating = 4, Walking = 5, Standing = 6, On Table = 7, Driving = 8
label_time_traces[label_time_traces == 0] = 0
label_time_traces[label_time_traces == '0'] = 'unknown'
label_time_traces[label_time_traces == 'unknown'] = 9
label_time_traces[label_time_traces == 'Washing Han'] = 1
label_time_traces[label_time_traces == 'Sitting'] = 2
label_time_traces[label_time_traces == 'Running'] = 3
label_time_traces[label_time_traces == 'Eating'] = 4
label_time_traces[label_time_traces == 'Walking'] = 5
label_time_traces[label_time_traces == 'Standing'] = 6
label_time_traces[label_time_traces == 'On Table'] = 7
label_time_traces[label_time_traces == 'Driving'] = 8

concat_label_time_traces = []
for j in range(len(label_time_traces)):
    concat_label_time_traces = np.concatenate((concat_label_time_traces,label_time_traces[j]),axis=0)

#np.save('concat_label_time_traces.npy',concat_label_time_traces)
plt.plot(concat_label_time_traces,'ro')
plt.text(118000, 8, r'$1:Washing Hands, 2:Sitting, 3:Running, 4:Eating$')
plt.text(118000, 7, r'$5:Walking, 6:Standing, 7:On Table, 8:Driving$')
plt.show()