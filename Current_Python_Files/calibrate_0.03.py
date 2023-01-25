## Note to Self
## Py = MAT - 1
## Py            MAT
## Range(A,B) == A:(B-1)

import time
st = time.process_time()

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import statistics as stat
import scipy as scp
import sympy as sym

print("Program to Calibrate IMU Gyroscope and Accelerometer")
    
def var(data):
    n = len(data)
    mean = sum(data) / n
    return sum((x - mean)**2 for x in data)/(n-1)

def accCostFunctionSQNONLIN(E,a_hat):
    misalignmentMatrix = np.array([[1,-E[0],E[1]],[0,1,-E[2]],[0,0,1]])
    scalingMtrix = np.diag([E[3],E[4],E[5]])

    a_bar = misalignmentMatrix @ scalingMtrix @ selectedAccData - (np.diag([E[6],E[7],E[8]]) @ np.ones([3,a_hat.shape[1]]))

    # Magnitude taken from tables
    magnitude = 9.81744

    residuals = np.zeros([a_bar.shape[1]])

    for i in range(a_bar.shape[1]):
        residuals[i] = magnitude**2 - (a_bar[0,i]**2+a_bar[1,i]**2+a_bar[2,i])

    return residuals

## Importing Data
print('Importing Data')
alpha_data_file = pd.read_excel('.\Data\IMU0x2Dalpha.xlsx')
alphadata = alpha_data_file.to_numpy()
omega_data_file = pd.read_excel('.\Data\IMU0x2Domega.xlsx')
omegadata = omega_data_file.to_numpy()
print(alphadata)
print('Data Import Complete')

total_time = alphadata[:,0]
T_init = 3000   # Initilization Time (s/100)

alpha_x = alphadata[:,1]
alpha_y = alphadata[:,2]
alpha_z = alphadata[:,3]
omega_x = omegadata[:,1]
omega_y = omegadata[:,2]
omega_z = omegadata[:,3]

bias_alpha_x = 33123
bias_alpha_y = 33276
bias_alpha_z = 32360

bias_omega_x = 32768
bias_omega_y = 32466
bias_omega_z = 32485

biasfree_alpha_x = alpha_x - bias_alpha_x
biasfree_alpha_y = alpha_y - bias_alpha_y
biasfree_alpha_z = alpha_z - bias_alpha_z

biasfree_omega_x = omega_x - bias_omega_x
biasfree_omega_y = omega_y - bias_omega_y
biasfree_omega_z = omega_z - bias_omega_z

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
    
normal_x = np.zeros([1,total_sample])
normal_y = np.zeros([1,total_sample])
normal_z = np.zeros([1,total_sample])

print('a')
for i in range (half_tw+1,total_sample-(half_tw)):
    normal_x[0,i-1] = var(biasfree_alpha_x[i-half_tw-1:i+half_tw])
    normal_y[0,i-1] = var(biasfree_alpha_y[i-half_tw-1:i+half_tw])
    normal_z[0,i-1] = var(biasfree_alpha_z[i-half_tw-1:i+half_tw])
print('b')
s_square = (normal_x**2)+(normal_y**2)+(normal_z**2)
print(s_square[0,1000])
print(normal_x[0,996:1003])

s_filter = np.zeros([1,total_sample])
print('finished')

##plt.plot(variance_magnitude)
##plt.show()

## Cycle used to individuate the optimal threshold

max_times_the_var = 10
res_norm_vector = np.zeros([9+1+1,max_times_the_var])

for times_the_var in range (1,max_times_the_var+1):
    print(times_the_var)
    for i in range (half_tw, total_sample - (half_tw)):
        if s_square[0,i-1] < times_the_var*var_3D:
            s_filter[0,i-1] = 1

    ffilter = s_filter
    L = 0
    samples = 0
    start = 0
    flag = 0

    for f in range (ffilter.shape[1]):
        if flag == 1 and ffilter[0,f] == 0:
            L += 1
            flag = 0
        elif flag == 0 and ffilter[0,f] == 1:
            flag = 1

    QS_time_interval_info_matrix = np.zeros([L,1+1+1])
    L = 0

    if ffilter[0,0] == 0:
        flag = 0
    else:
        flag = 1
        start = 1

    # Cycle to determine the QS_time_interval_info_matrix
    for i in range (ffilter.shape[1]):
##        print(i)
        if flag == 1 and ffilter[0,i] == 1:
            samples += 1
        elif flag == 1 and ffilter[0,i] == 0:
            QS_time_interval_info_matrix[L,:] = [start,i,samples]
            L += 1
            flag = 0
        elif flag == 0 and ffilter[0,i] == 1:
            start = i + 1
            samples = 1
            flag = 1

    # data selection - accelerometer
    qsTime = 1
    sample_period = 0.01
    num_samples = int(qsTime/sample_period)

    signal = np.array([biasfree_alpha_x,biasfree_alpha_y,biasfree_alpha_z])

    L = 0

    for j in range (len(QS_time_interval_info_matrix)):
        if QS_time_interval_info_matrix[j,2] >= num_samples:
            for i in range(num_samples-1):
                L += 1
            L += 1

    selected_data = np.zeros([3,L])
    L = 0

    for j in range (len(QS_time_interval_info_matrix)):
        if QS_time_interval_info_matrix[j,2] >= num_samples:
            selection_step = QS_time_interval_info_matrix[j,2]//num_samples
            for i in range(num_samples-1):
                selected_data[0,L] = signal[0,int(QS_time_interval_info_matrix[j,0] + (i)*selection_step)-1]
                selected_data[1,L] = signal[1,int(QS_time_interval_info_matrix[j,0] + (i)*selection_step)-1]
                selected_data[2,L] = signal[2,int(QS_time_interval_info_matrix[j,0] + (i)*selection_step)-1]
                L += 1
            selected_data[0,L] = signal[0,int(QS_time_interval_info_matrix[j,1])-1]
            selected_data[1,L] = signal[1,int(QS_time_interval_info_matrix[j,1])-1]
            selected_data[2,L] = signal[2,int(QS_time_interval_info_matrix[j,1])-1]
            L += 1

    # minimization
    selectedAccData = selected_data

##    theta_pr = np.zeros(9)

    def fun(E):
        return accCostFunctionSQNONLIN(E,selectedAccData)

    theta_pr = np.array([0,0,0,1,1,1,0,0,0])#np.zeros(9)

    output = scp.optimize.least_squares(fun,theta_pr,ftol=1e-10,method='lm',max_nfev=150000)

    for i in range(len(output.x)):
        res_norm_vector[i,times_the_var-1] = output.x[i]
    res_norm_vector[9,times_the_var-1] = sum(output.fun**2)
    res_norm_vector[10,times_the_var-1] = times_the_var*var_3D

print(res_norm_vector[:,0])

time = time.process_time() - st
print('Execution TIme:',time,'seconds')
