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

print("Program to Calibrate IMU Gyroscope and Accelerometer")

def var(data):
    n = len(data)
    mean = sum(data) / n
    return sum((x - mean)**2 for x in data)/(n-1)

def var_tw(tw,datax,datay,dataz):
    total_sample = len(biasfree_omega_x)
    half_tw = tw//2
    
    normal_x = np.zeros(total_sample)
    normal_y = np.zeros(total_sample)
    normal_z = np.zeros(total_sample)
    print('a')
    for i in range (half_tw+1,total_sample-(half_tw+1)):
        normal_x[i] = var(datax[i-half_tw:i+half_tw])
        normal_y[i] = var(datay[i-half_tw:i+half_tw])
        normal_z[i] = var(dataz[i-half_tw:i+half_tw])
    print('b')
    s_square = (normal_x**2)+(normal_y**2)+(normal_z**2)
    print(s_square[1000])
    return s_square

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

bias_omega_x = np.mean(omega_x[0:T_init])
bias_omega_y = np.mean(omega_y[0:T_init])
bias_omega_z = np.mean(omega_z[0:T_init])

biasfree_alpha_x = alpha_x - bias_alpha_x
biasfree_alpha_y = alpha_y - bias_alpha_y
biasfree_alpha_z = alpha_z - bias_alpha_z

biasfree_omega_x = omega_x - bias_omega_x
biasfree_omega_y = omega_y - bias_omega_y
biasfree_omega_z = omega_z - bias_omega_z

plt.plot(omegadata[:,0],biasfree_omega_x,label = 'x Axis')
plt.plot(omegadata[:,0],biasfree_omega_y,label = 'y Axis')
plt.plot(omegadata[:,0],biasfree_omega_z,label = 'z Axis')
plt.title('Bias Free Gyroscope Data')
plt.xlabel('Time (s/100)')
plt.ylabel('Raw Gyroscope')
plt.legend()
plt.show()

M_inf = []

print('calculating variance')
var_3D = (var(biasfree_alpha_x[0:T_init]))**2+(var(biasfree_alpha_y[0:T_init]))**2+(var(biasfree_alpha_z[0:T_init]))**2
variance_magnitude = np.sqrt(var_tw(101,biasfree_alpha_x,biasfree_alpha_y,biasfree_alpha_z))
print('finished')

plt.plot(variance_magnitude)
plt.show()



for i in range (1,10):
    print(i)

time = time.process_time() - st
print('Execution TIme:',time,'seconds')
