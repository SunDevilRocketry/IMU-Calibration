import time
st = time.process_time()

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import statistics as stat
##from statistics import variance as var
import scipy as scp
import sympy as sym
from array import *

print("Program to Calibrate IMU Gyroscope and Accelerometer")
    
def var(data):
    n = len(data)
    mean = sum(data) / n
    return sum((x - mean)**2 for x in data)/(n-1)

def mean(data):
    n = len(data)
    mean = sum(data) / n

##def zeros(rows,columns):
##    if rows =0 1 or rows == 0:
##        return [0 for row in range(rows)]
##    else:
##        return [[0 for col in range(columns)] for row in range(rows)]

class zeros:
    def matrix(rows,columns):
        return [[0 for col in range(columns)] for row in range(rows)]
    def row(rows):
        return [0 for row in range(rows)]

## Importing Data

print('Importing Data')
alpha_data_file = pd.read_excel('.\Data\IMU0x2Dalpha.xlsx')
alphadata = alpha_data_file.to_numpy().tolist()
omega_data_file = pd.read_excel('.\Data\IMU0x2Domega.xlsx')
omegadata = omega_data_file.to_numpy().tolist()
##print(alphadata)
print('Data Import Complete')

total_sample = len(omegadata)
T_init = 3000 # Initilization Time (s/100)

total_time = zeros.row(total_sample)
alpha_x = zeros.row(total_sample)
alpha_y = zeros.row(total_sample)
alpha_z = zeros.row(total_sample)
omega_x = zeros.row(total_sample)
omega_y = zeros.row(total_sample)
omega_z = zeros.row(total_sample)

for i in range (total_sample):
    total_time[i] = alphadata[i][0]
    alpha_x[i] = alphadata[i][1]
    alpha_y[i] = alphadata[i][2]
    alpha_z[i] = alphadata[i][3]
    omega_x[i] = omegadata[i][1]
    omega_y[i] = omegadata[i][2]
    omega_z[i] = omegadata[i][3]

bias_alpha_x = 33123
bias_alpha_y = 33276
bias_alpha_z = 32360

bias_omega_x = mean(omega_x[0:T_init])
bias_omega_y = mean(omega_y[0:T_init])
bias_omega_z = mean(omega_z[0:T_init])

for i in range (total_sample):
    biasfree_alpha_x[i] = alpha_x[0][i] - bias_alpha_x
    biasfree_alpha_y[i] = alpha_y[0][i] - bias_alpha_y
    biasfree_alpha_z[i] = alpha_z[0][i] - bias_alpha_z
    biasfree_omega_x[i] = omega_x[0][i] - bias_omega_x
    biasfree_omega_y[i] = omega_y[0][i] - bias_omega_y
    biasfree_omega_z[i] = omega_z[0][i] - bias_omega_z

total_sample = len(biasfree_omega_x)

##plt.plot(omegadata[:,0],biasfree_omega_x,label = 'x Axis')
##plt.plot(omegadata[:,0],biasfree_omega_y,label = 'y Axis')
##plt.plot(omegadata[:,0],biasfree_omega_z,label = 'z Axis')
##plt.title('Bias Free Gyroscope Data')
##plt.xlabel('Time (s/100)')
##plt.ylabel('Raw Gyroscope')
##plt.legend()
##plt.show()

M_inf = []

## Static State Statistical Filter

print('calculating variance')
var_3D = (var(biasfree_alpha_x[0:T_init]))**2 + (var(biasfree_alpha_y[0:T_init]))**2 + (var(biasfree_alpha_z[0:T_init]))**2

tw = 101
half_tw = tw//2
    
normal_x = zeros(1,total_sample)
normal_y = zeros(1,total_sample)
normal_z = zeros(1,total_sample)
s_square = zeros(1,total_sample)

print('a')
for i in range (half_tw+1,total_sample-(half_tw+1)):
    normal_x[0][i] = var(biasfree_alpha_x[i-half_tw:i+half_tw])
    normal_y[0][i] = var(biasfree_alpha_y[i-half_tw:i+half_tw])
    normal_z[0][i] = var(biasfree_alpha_z[i-half_tw:i+half_tw])
    s_square[0][i] = (normal_x[0][i]**2)+(normal_y[0][i]**2)+(normal_z[0][i]**2)
print('b')
##s_square = (normal_x**2)+(normal_y**2)+(normal_z**2)
print(s_square[0][1000])

s_filter = np.zeros(total_sample)
print('finished')

##plt.plot(variance_magnitude)
##plt.show()

## Cycle used to individuate the optimal threshold

max_times_the_var = 10
res_norm_vector = zeros(9+1+1,max_times_the_var)

for times_the_var in range (max_times_the_var):
    print(times_the_var)
    for i in range (half_tw, total_sample - (half_tw + 1)):
        if s_square[i] < times_the_var*var_3D:
            s_filter[i] = 1

    ffilter = s_filter
    l = 0
##    QS_time_interval_info_matrix = np.zeros(1+1+1)
    samples = 0
    start = 0

    flag = 0

    if ffilter[1] == 0:
        flag = 0
    else:
        flag = 1
        start = 1

    for i in range (len(ffilter)):
        if flag == 1 and ffilter[i] == 0:
            l += 1
    if l > 0:
        QS_time_interval_info_matrix = np.zeros([l,1+1+1])
        l = 0
        for i in range (len(ffilter)):
##            print(i)
            if flag == 1 and ffilter[i] == 1:
                samples += 1
            elif flag == 1 and ffilter[i] == 0:
                QS_time_interval_info_matrix[l,0] = start
                QS_time_interval_info_matrix[l,1] = i-1
                QS_time_interval_info_matrix[l,2] = samples
                l += 1
                flag = 0
            elif flag == 0 and ffilter[i] == 1:
                start = i
                samples = 1
                flag = 1
    else:
        QS_time_interval_info_matrix = np.zeros([1+1+1])
        l = 0
        for i in range (len(ffilter)):
##            print(i)
            if flag == 1 and ffilter[i] == 1:
                samples += 1
            elif flag == 1 and ffilter[i] == 0:
                QS_time_interval_info_matrix[0] = start
                QS_time_interval_info_matrix[1] = i-1
                QS_time_interval_info_matrix[2] = samples
                l += 1
                flag = 0
            elif flag == 0 and ffilter[i] == 1:
                start = i
                samples = 1
                flag = 1
##    l = 0

##    # Cycle to determine the QS_time_interval_info_matrix
##    for i in range (1, len(ffilter)):
##        print(i)
##        if flag == 1 and ffilter[i] == 1:
##            samples += 1
##        elif flag == 1 and ffilter[i] == 0:
##            QS_time_interval_info_matrix[l,0] = start
##            QS_time_interval_info_matrix[l,1] = i-1
##            QS_time_interval_info_matrix[l,2] = samples
##            l += 1
##            flag = 0
##        elif flag == 0 and ffilter[i] == 1:
##            start = i
##            samples = 1
##            flag = 1

    # data selection - accelerometer
    qsTime = 1
    sample_period = 0.01
    num_samples = qsTime/sample_period
    signal = np.zeros([len(biasfree_alpha_x),3])

    for i in range (len(biasfree_alpha_x)):
        signal[i,0] = biasfree_alpha_x[i]
        signal[i,1] = biasfree_alpha_y[i]
        signal[i,2] = biasfree_alpha_z[i]

    selected_data = np.zeros([3,1])
    l = 1

##    for j in range (QS_time_interval_info_matrix[:,1]):
##        k = 0

time = time.process_time() - st
print('Execution TIme:',time,'seconds')
