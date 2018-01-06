# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:08:19 2017

@author: Gordon
"""

import csv
import glob
import numpy as np
import pylab
#data_dir = "../Sample Dataset-20170228T035448Z-001/Sample Dataset/February 2016 Sample (old)/AlgoSnap Sample Dataset CSV Format/AlgoSnap Sample Dataset CSV Format/participant-one-csv/0_Accelerometer-352622063881655_1454956346105.csv"

from datetime import datetime

'''
Accelerometer (Smart phone)
'''
Accelerometer_list = glob.glob("0_Accelerometer-3*.csv")

Accelerometer_trace = []
for i in range(len(Accelerometer_list)):
    with open(Accelerometer_list[i], newline='') as csvfile:
        Accelerometer_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        next(Accelerometer_reader, None)
        #next(csvfile)
        Accelerometer = []
        
        for row in Accelerometer_reader:
            
            row_split = row[0].split(",")
            dt = datetime.fromtimestamp(int(row_split[2]) // 1000000000)
            s = dt.strftime('%Y-%m-%d %H:%M:%S')
            #s += '.' + str(int(1454956105656222395 % 1000000000)).zfill(9)
            row_split[2] = s
            Accelerometer.append(row_split)
            #print(', '.join(row))
        #del Accelerometer[0]
    Accelerometer_trace.append(Accelerometer)

    
#dt = datetime.fromtimestamp(1454956105656222395 // 1000000000)
#s = dt.strftime('%Y-%m-%d %H:%M:%S')
#s += '.' + str(int(1454956105656222395 % 1000000000)).zfill(9)

#Battery_list = glob.glob("0_Battery*.csv")
#
#Battery_trace = []
#for i in range(len(Battery_list)):
#    with open(Battery_list[i], newline='') as csvfile:
#        Battery_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#        Battery = []
#        for row in Battery_reader:
#            row_split = row[0].split(",")
#            Battery.append(row_split)
#            #print(', '.join(row))
#        del Battery[0]
#    Battery_trace.append(Battery)
#    
Bluetooth_list = glob.glob("0_Bluetooth-3*.csv")

Bluetooth_trace = []
for i in range(len(Bluetooth_list)):
    with open(Bluetooth_list[i], newline='') as csvfile:
        Bluetooth_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        next(Bluetooth_reader, None)
        Bluetooth = []
        for row in Bluetooth_reader:
            row_split = row[0].split(",")
            dt = datetime.fromtimestamp(int(row_split[2]) // 1000000000)
            s = dt.strftime('%Y-%m-%d %H:%M:%S')
            row_split[2] = s
            Bluetooth.append(row_split)
            #print(', '.join(row))
        #del Bluetooth[0]
    Bluetooth_trace.append(Bluetooth)
#    
#    
#Connectivity_list = glob.glob("0_Connectivity*.csv")
#    
#Connectivity_trace = []
#for i in range(len(Connectivity_list)):
#    with open(Connectivity_list[i], newline='') as csvfile:
#        Connectivity_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#        Connectivity = []
#        for row in Connectivity_reader:
#            row_split = row[0].split(",")
#            Connectivity.append(row_split)
#            #print(', '.join(row))
#        del Connectivity[0]
#    Connectivity_trace.append(Connectivity)
#    
#GSM_list = glob.glob("0_GSM*.csv")
#    
#GSM_trace = []
#for i in range(len(GSM_list)):
#    with open(GSM_list[i], newline='') as csvfile:
#        GSM_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#        GSM = []
#        for row in GSM_reader:
#            row_split = row[0].split(",")
#            GSM.append(row_split)
#            #print(', '.join(row))
#        del GSM[0]
#    GSM_trace.append(GSM)


#IntervalLabel_list = glob.glob("0_IntervalLabel-3*.csv")
#
#IntervalLabel_trace = []
#for i in range(len(IntervalLabel_list)):
#    with open(IntervalLabel_list[i], newline='') as csvfile:
#        IntervalLabel_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#        next(IntervalLabel_reader, None)
#        IntervalLabel = []
#        for row in IntervalLabel_reader:
#            row_split = row[0].split(",")
#            dt = datetime.fromtimestamp(int(row_split[3]) // 1000000000)
#            s = dt.strftime('%Y-%m-%d %H:%M:%S')
#            row_split[3] = s
#            dt = datetime.fromtimestamp(int(row_split[5]) // 1000000000)
#            s = dt.strftime('%Y-%m-%d %H:%M:%S')
#            row_split[5] = s
#            IntervalLabel.append(row_split)
#            #print(', '.join(row))
#        #del IntervalLabel[0]
#    IntervalLabel_trace.append(IntervalLabel)
    
Location_list = glob.glob("0_Location-3*.csv")

Location_trace = []
for i in range(len(Location_list)):
    with open(Location_list[i], newline='') as csvfile:
        Location_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        next(Location_reader, None)
        Location = []
        for row in Location_reader:
            row_split = row[0].split(",")
            dt = datetime.fromtimestamp(int(row_split[2]) // 1000000000)
            s = dt.strftime('%Y-%m-%d %H:%M:%S')
            row_split[2] = s
            Location.append(row_split)
            #print(', '.join(row))
        #del Location[0]
    Location_trace.append(Location)

    
## Location
from math import radians, cos, sin, asin, sqrt, atan2


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    c_new = 2 * atan2(sqrt(a), sqrt(1-a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    km = c_new * r
    return km
    

    
lat1 = float(Location_trace[0][0][3])
lat2 = float(Location_trace[-1][0][3])
lon1 = float(Location_trace[0][0][4])
lon2 = float(Location_trace[-1][0][4])
#lat1 = 53.32055555555556
#lat2 = 53.31861111111111
#lon1 = -1.7297222222222221
#lon2 = -1.6997222222222223
    
distance = haversine(lon1, lat1, lon2, lat2)


### Accelerometer
x = float(Accelerometer_trace[0][0][3])
y = float(Accelerometer_trace[0][0][4])
z = float(Accelerometer_trace[0][0][5])

accel_magnitude = sqrt(x**2 + y**2 + z**2)

#To underestimate haversine(lat1, long1, lat2, long2) * 0.90 or whatever factor you want. I don't see how introducing error to your underestimation is useful.

### Smoothing for bluetooth
data = [0]*30

data[3]=1
data[12]=1
data[15]=1
data[25]=1

def smoothListGaussian(myarray, degree=5):
    myarray = np.pad(myarray, (degree-1,degree-1), mode='edge')
    window = degree*2-1
    weight = np.arange(-degree+1, degree)/window
    weight = np.exp(-(16*weight**2))
    weight /= sum(weight)
    smoothed = np.convolve(myarray, weight, mode='valid')
    return smoothed

sLG = smoothListGaussian(data)

pylab.figure(figsize=(550/80,700/80))  

pylab.suptitle('1D Data Smoothing', fontsize=16)  

pylab.subplot(2,1,1)  

p1=pylab.plot(data,".k")  

p1=pylab.plot(data,"-k")  

a=pylab.axis()  

pylab.axis([a[0],a[1],-.1,1.1])  

pylab.text(2,.8,"raw data",fontsize=14)  


pylab.subplot(2,1,2)  

p1=pylab.plot(smoothListGaussian(data),".k")  

p1=pylab.plot(smoothListGaussian(data),"-k")  

pylab.axis([a[0],a[1],-.1,.4])  

pylab.text(2,.3,"gaussian smoothing",fontsize=14)