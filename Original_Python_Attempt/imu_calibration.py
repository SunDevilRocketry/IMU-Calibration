import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statistics import variance as var
import scipy as scp
from accCostFunctLSQNONLIN import *
import matplotlib as mpl
import sympy as sym

print("All Imported")

print("Program to Calibrate IMU Gyroscope and Accelerometer")

print("Data Import")
alphadata = scp.io.loadmat(r"C:\Users\admin\OneDrive - Arizona State University\College\Rocketry\IMU calibration\problem\IMU0x2Dalpha.mat")
IMU0x2Dalpha = alphadata['IMU0x2Dalpha']
omegadata = scp.io.loadmat(r"C:\Users\admin\OneDrive - Arizona State University\College\Rocketry\IMU calibration\problem\IMU0x2Domega.mat")
IMU0x2Domega = omegadata['IMU0x2Domega']
print(IMU0x2Dalpha)

time = IMU0x2Domega[:,0]

intervals = np.zeros(len(time))
intervals[0] = ((time[1] - time[0])/2) + time[0]/2
intervals[1:len(intervals)] = time[1:len(time)] - time[0:len(time)-1]

total_sample = len(time)

offset_acc = [33123,33276,32360]
offset_gyro = [32768,32466,32485]

a_xp = IMU0x2Dalpha[:,1] - offset_acc[0]
a_yp = IMU0x2Dalpha[:,2] - offset_acc[1]
a_zp = IMU0x2Dalpha[:,3] - offset_acc[2]
omega_x = IMU0x2Domega[:,1] - offset_gyro[0]
omega_y = IMU0x2Domega[:,2] - offset_gyro[1]
omega_z = IMU0x2Domega[:,3] - offset_gyro[2]

print("Static State Statistical Filter")

var_3D = (var(a_xp[0:3000])**2 + var(a_yp[0:3000])**2 + var(a_zp[0:3000])**2)

w_d = 101           # Window's dimension

normal_x = np.zeros(total_sample)
normal_y = np.zeros(total_sample)
normal_z = np.zeros(total_sample)

half_w_d = w_d//2

print("initialize integral")
integral_x = sum(a_xp[0:w_d]*0.01)
integral_y = sum(a_yp[0:w_d]*0.01)
integral_z = sum(a_zp[0:w_d]*0.01)

for i in range (half_w_d+1,total_sample-(half_w_d+1)):
    normal_x[i] = var(a_xp[i-(half_w_d):i+half_w_d])
    normal_y[i] = var(a_yp[i-(half_w_d):i+half_w_d])
    normal_z[i] = var(a_zp[i-(half_w_d):i+half_w_d])

s_square = normal_x**2+normal_y**2+normal_z**2

print('sum(normal_x) = ' + str(sum(normal_x)))
print('sum(normal_y) = ' + str(sum(normal_y)))
print('sum(normal_z) = ' + str(sum(normal_z)))

#input("Press Enter To Continue")

plt.plot(s_square)
plt.show()

s_filter = np.zeros(total_sample)

print("Cycle used to individuate the optimal threshold")

max_times_the_var = 10

res_norm_vector = np.zeros((9+1+1,max_times_the_var))

for times_the_var in range (1,max_times_the_var+1):
    for i in range (half_w_d,total_sample-(half_w_d+1)):
        if s_square[i] < times_the_var*var_3D:
            s_filter[i] = 1

    ffilter = s_filter
    l = 0
    QS_time_interval_info_matrix = np.empty((0,3))
    samples = 0
    start = 0

    falg = 0

    if ffilter[0] == 0:
        flag = 0
    else:
        flag = 1
        start = 1

    print("cycle to determine the QS_time_interval_info_matrix")
    for i in range (len(ffilter)):
        if flag == 0 and ffilter[i] == 0:
            d = 1
        elif flag == 1 and ffilter[i] == 1:
            samples += 1
        elif flag == 1 and ffilter[i] == 0:
            QS_time_interval_info_matrix = np.r_[QS_time_interval_info_matrix,[np.array([start,i-1,samples])]]
            #np.append(QS_time_interval_info_matrix,np.array([start,i-1,samples]))
            l += 1
            flag = 0
        elif flag == 0 and ffilter[i] == 1:
            start = i
            samples = 1
            flag = 1
    #QS_time_interval_info_matrix = QS_time_interval_info_matrix[1:len(QS_time_interval_info_matrix)-1,:]

    print("data selection - accelerometer")
    qsTime = 1
    sample_period = 0.01
    num_samples = int(qsTime/sample_period)

    signal = np.matrix([a_xp,a_yp,a_zp])

    selected_data = [[],[],[]]
    l = 0

    for j in range (len(QS_time_interval_info_matrix)):#[:,0])):
        if QS_time_interval_info_matrix[j,2] < num_samples:
            d = 1
        else:
            selection_step = QS_time_interval_info_matrix[j,2]//num_samples
            for i in range (num_samples-1):
                selected_data = np.r_['1,2,0',selected_data,signal[:, int(QS_time_interval_info_matrix[j,0] + (i)*selection_step)]]
                l += 1
            selected_data = np.r_['1,2,0',selected_data,signal[:,int(QS_time_interval_info_matrix[j,1])]]
            l += 1

    print("minimization")
    selectedAccData = selected_data

    theta_pr = np.zeros(9)
##    #############
##    #accCostFunctLSQNONLIN(theta_pr,selectedAccData)
##    E = sym.Symbol('theta_pr')
##    a_hat = selectedAccData
##
##    misalignmentMatrix = np.array([[1, -E[0], E[1]], [0, 1, -E[2]], [0, 0, 1]])
##    scalingMtrix = np.diag([E[4], E[5], E[6]])
##
##    a_bar = misalignmentMatrix*scalingMtrix*(a_hat) - (np.diag([E[7],E[8],E[9]])*np.ones([3,a_hat.shape[1]]))
##
##    # Magnitude taken from tables
##    magnitude = 9.81744
##
##    residuals = np.zeros([a_hat.shape[1],1])
##
##    for i in range (a_hat.shape[1]):
##        residuals[i,0] = magnitude**2 - (a_bar[0,i]**2 + a_bar[1,i]**2 + a_bar[2,i]**2)
##
##    res_vector = residuals
##    #############

##    options =

##    scp.optimize.least_squares
print(QS_time_interval_info_matrix)
