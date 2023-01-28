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

    a_bar = np.matmul(np.matmul(misalignmentMatrix,scalingMtrix),a_hat) - np.matmul(np.diag([E[6],E[7],E[8]]),np.ones([3,a_hat.shape[1]]))

    # Magnitude taken from tables
    magnitude = 9.81744

    residuals = np.zeros([a_bar.shape[1]])

    for i in range(a_bar.shape[1]):
        residuals[i] = magnitude**2 - (a_bar[0,i]**2 + a_bar[1,i]**2 + a_bar[2,i]**2)

    return residuals

def gyroCostFunctLSQNONLINUsingOnlyTheFilter(E, QS_time_interval_info_matrix, omega_x_hat, omega_y_hat, omega_z_hat):

    omega_hat = np.array([omega_x_hat,omega_y_hat,omega_z_hat]])
    
    misalignmentMatrix = np.array([[1,E[1],E[2]],[E[3],1,E[5]],[E[6],E[7],1]])
    scalingMatrix = np.diag([E[0],E[4],E[8]])

    omega_bar = np.matmul(np.matmul(misalignmentMatrix,scalingMatrix),omega_hat)

    omega_x = omega_bar[0,:]
    omega_y = omega_bar[1,:]
    omega_z = omega_bar[2,:]

    vector = np.zeros([3,5])

    for pr in range (QS_time_interval_info_matrix.shape[1]-1)
    
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
n = 16
y = 6
r = (2**n - 1)/(2*y)

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

##plt.plot(total_time,biasfree_omega_x,label = 'x Axis')
##plt.plot(total_time,biasfree_omega_y,label = 'y Axis')
##plt.plot(total_time,biasfree_omega_z,label = 'z Axis')
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

    theta_pr = np.array([0,0,0,0,0,0,0,0,0])#np.zeros(9)

    output = scp.optimize.least_squares(fun,theta_pr,ftol=1e-10,method='lm',max_nfev=150000)

    for i in range(len(output.x)):
        res_norm_vector[i,times_the_var-1] = output.x[i]
    res_norm_vector[9,times_the_var-1] = sum(output.fun**2)
    res_norm_vector[10,times_the_var-1] = times_the_var*var_3D
# End main for loop
print(res_norm_vector[:,0])

vec = res_norm_vector[9,:]

z = np.where(vec == min(vec))[0][0]

threshold_opt = res_norm_vector[10,z]

theta_pr_opt = res_norm_vector[0:9,z]

estimated_misalignmentMatrix = np.array([[1,-theta_pr_opt[0],theta_pr_opt[1]],[0,1,-theta_pr_opt[2]],[0,0,1]])
estimated_scalingMatrix = np.diag([theta_pr_opt[3],theta_pr_opt[4],theta_pr_opt[5]])
estimated_biasVector = [[theta_pr_opt[6]],[theta_pr_opt[7]],[theta_pr_opt[8]]]

s_filter = np.zeros([total_sample])

for i in range (half_tw, total_sample - (half_tw)):
    if s_square[0,i-1] < times_the_var*var_3D:
        s_filter[i-1] = 1

##plt.plot(total_time,biasfree_alpha_x,'r-',label = 'x Axis')
##plt.plot(total_time,biasfree_alpha_y,'g-',label = 'y Axis')
##plt.plot(total_time,biasfree_alpha_z,'b-',label = 'z Axis')
##plt.plot(total_time,s_filter*5000,'k-')
##plt.xlabel('Time (s/100)')
##plt.ylabel('Raw Accelerometer')
##plt.legend()
##plt.show()

## GYROSCOPE BIAS REMOVAL
# QS filter for the first static region individuation
init_long_qs_interval_start = 0
init_long_qs_interval_end = 0
flag_is_first_long_static_interval = 1

for i in range (total_sample):
    if s_filter[i] == 1 and flag_is_first_long_static_interval == 1:
        init_long_qs_interval_start = i+1
        flag_is_first_long_static_interval = 2
    elif s_filter[i] == 0 and flag_is_first_long_static_interval == 2:
        init_long_qs_interval_end = i+1
        break

estimate_bias_x = np.mean(biasfree_omega_x[init_long_qs_interval_start:init_long_qs_interval_end])
estimate_bias_y = np.mean(biasfree_omega_y[init_long_qs_interval_start:init_long_qs_interval_end])
estimate_bias_z = np.mean(biasfree_omega_z[init_long_qs_interval_start:init_long_qs_interval_end])

omega_x = biasfree_omega_x - estimate_bias_x
omega_y = biasfree_omega_y - estimate_bias_y
omega_z = biasfree_omega_z - estimate_bias_z

## GYROSCOPE MINIMIZATION
estimated_acc_misalignmentMatrix = estimated_misalignmentMatrix
estimated_acc_scalingMatrix = estimated_scalingMatrix
estimated_acc_biasVector = estimated_biasVector
# Calibrating accelerometer data
A = np.matmul(np.matmul(estimated_misalignmentMatrix,estimated_acc_scalingMatrix),np.array([biasfree_alpha_x,biasfree_alpha_y,biasfree_alpha_z]))
B = np.matmul(np.diag([estimated_biasVector[0][0],estimated_biasVector[1][0],estimated_biasVector[2][0]]),np.ones([3,len(biasfree_alpha_x)]))
calib_acc = A - B
alpha_x_calib = calib_acc[0,:]
alpha_y_calib = calib_acc[1,:]
alpha_z_calib = calib_acc[2,:]

ffilter = s_filter
ffilter = ffilter.reshape([1,len(s_filter)])
L = 0
samples = 0
start = 0
# Inizializzazione flag
flag = 0

for f in range (ffilter.shape[1]):
    if flag == 1 and ffilter[0,f] == 0:
        L += 1
        flag = 0
    elif flag == 0 and ffilter[0,f] == 1:
        flag = 1

QS_time_interval_info_matrix = np.zeros([1+1+1+3,L])
L = 0

if ffilter[0,0] == 0:
    flag = 0
else:
    flag = 1
    start = 1

# Cycle
for i in range (ffilter.shape[1]):
##        print(i)
    if flag == 1 and ffilter[0,i] == 1:
        samples += 1
    elif flag == 1 and ffilter[0,i] == 0:
        QS_time_interval_info_matrix[0:3,L] = [start,i,samples]
        L += 1
        flag = 0
    elif flag == 0 and ffilter[0,i] == 1:
        start = i + 1
        samples = 1
        flag = 1

num_samples = int(qsTime/sample_period)

signal = np.array([alpha_x_calib,alpha_y_calib,alpha_z_calib])

L = 0

for j in range (len(QS_time_interval_info_matrix)):
    for i in range(num_samples-1):
        L += 1
    L += 1

selected_data = np.zeros([3,L])
L = 0

for j in range (len(QS_time_interval_info_matrix)):
    selected_acc_data = np.zeros([3,num_samples-1])
    selection_step = QS_time_interval_info_matrix[2,j]//num_samples
    for i in range(num_samples-1):
        selected_acc_data[0,i] = signal[0,int(QS_time_interval_info_matrix[0,j] + (i)*selection_step)-1]
        selected_acc_data[1,i] = signal[1,int(QS_time_interval_info_matrix[0,j] + (i)*selection_step)-1]
        selected_acc_data[2,i] = signal[2,int(QS_time_interval_info_matrix[0,j] + (i)*selection_step)-1]
##            L += 1
    selected_data[0,num_samples-1] = signal[0,int(QS_time_interval_info_matrix[1,j])-1]
    selected_data[1,num_samples-1] = signal[1,int(QS_time_interval_info_matrix[1,j])-1]
    selected_data[2,num_samples-1] = signal[2,int(QS_time_interval_info_matrix[1,j])-1]
    QS_time_interval_info_matrix[3:6, L] = np.mean(selected_acc_data, axis=1)
    L += 1

QS_time_interval_calib_info_matrix = QS_time_interval_info_matrix

# Minimizing LSQNONLIN


time = time.process_time() - st
print('Execution TIme:',time,'seconds')
