import numpy as np
import math

def accCostFunctionLSQNONLIN(E,a_hat):
    misalignmentMatrix = np.array([[1,-E[0],E[1]],[0,1,-E[2]],[0,0,1]])
    scalingMtrix = np.diag([E[3],E[4],E[5]])

    a_bar = np.matmul(np.matmul(misalignmentMatrix,scalingMtrix),a_hat) - np.matmul(np.diag([E[6],E[7],E[8]]),np.ones([3,a_hat.shape[1]]))

    # Magnitude taken from tables
    magnitude = 9.81744

    residuals = np.zeros([a_bar.shape[1]])

    for i in range(a_bar.shape[1]):
        residuals[i] = magnitude**2 - (a_bar[0,i]**2 + a_bar[1,i]**2 + a_bar[2,i]**2)

    return residuals

def gyroCostFunctionLSQNONLIN(E, QS_time_interval_info_matrix, omega_x_hat, omega_y_hat, omega_z_hat):

    omega_hat = np.array([omega_x_hat,omega_y_hat,omega_z_hat])
    
    misalignmentMatrix = np.array([[1,E[1],E[2]],[E[3],1,E[5]],[E[6],E[7],1]])
    scalingMatrix = np.diag([E[0],E[4],E[8]])

    omega_bar = np.matmul(np.matmul(misalignmentMatrix,scalingMatrix),omega_hat)

    omega_x = omega_bar[0,:]
    omega_y = omega_bar[1,:]
    omega_z = omega_bar[2,:]

    vector = np.zeros([(QS_time_interval_info_matrix.shape[1]-1)*3,5])

    for pr in range (1,QS_time_interval_info_matrix.shape[1]-1):
        vector[(pr-1)*3+1,0] = QS_time_interval_info_matrix[3,pr]
        vector[pr*3-1,0] = QS_time_interval_info_matrix[4,pr]
        vector[pr*3,0] = QS_time_interval_info_matrix[5,pr]
        vector[(pr-1)*3+1,4] = QS_time_interval_info_matrix[3,pr+1]
        vector[pr*3-1,4] = QS_time_interval_info_matrix[4,pr+1]
        vector[pr*3,4] = QS_time_interval_info_matrix[5,pr+1]
        gyroUnbiasUncalibratedValues = np.array([omega_x[int(QS_time_interval_info_matrix[1,pr-1]):int(QS_time_interval_info_matrix[0,pr] - 2)],omega_y[int(QS_time_interval_info_matrix[1,pr-1]):int(QS_time_interval_info_matrix[0,pr] - 2)],omega_z[int(QS_time_interval_info_matrix[1,pr-1]):int(QS_time_interval_info_matrix[0,pr] - 2)]])
        R = rotationRK4(gyroUnbiasUncalibratedValues)

def rotationRK4(omega):
    # delta t
    dt = 0.01

    omega_x = omega[0,:]
    omega_y = omega[1,:]
    omega_z = omega[2,:]

    num_samples = len(omega_x)

    q_k = fromOmegaToQ(np.array([[omega_x[0]],[omega_y[0]],[omega_z[0]]]),np.array([[0.01]]))
    q_next_k = np.zeros([4,1])

    for i in range (1,num_samples):
        # first Runge-Kutta coefficient
        q_i_1 = q_k
        OMEGA_omega_t_k = np.array([[0,-omega_x[i],-omega_y[i],-omega_z[i]],
                                    [omega_x[i],0,omega_z[i],-omega_y[i]],
                                    [omega_y[i],-omega_z[i],0,omega_x[i]],
                                    [omega_z[i],omega_y[i],-omega_x[i],0]])
        k_1 = 0.5*OMEGA_omega_t_k*q_i_1

def fromOmegaToQ(omega,intervals):
    s = omega.shape
    angularRotation = np.zeros([1,s[1]])
    direction = np.zeros([3,s[1]])
    q = np.zeros([s[1],4])

    for i in range (s[1]):
        angularRotation[0,i] = (omega[0,i]**2 + omega[1,i]**2 * omega[2,i]**2)**0.5*intervals
        direction[:,i] = omega[:,i]*(intervals/angularRotation[0,i])
        direct = np.transpose(direction[:,i])
        q[i,:] = np.array([math.cos(angularRotation[0,i]/2),
                           math.sin(angularRotation[0,i]/2)*direct[0],
                           math.sin(angularRotation[0,i]/2)*direct[1],
                           math.sin(angularRotation[0,i]/2)*direct[2]])

    return q

def quaternion(theta,w):

    q = np.array([math.cos(theta/2),math.sin(theta/2)*w[0],math.sin(theta/2)*w[1],math.sin(theta/2)*w[2]])

    r_11 = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
    r_12 = 2*(q[1]*q[2] - q[0]*q[3])
    r_13 = 2*(q[1]*q[3] + q[0]*q[2])
    r_21 = 2*(q[1]*q[2] + q[0]*q[3])
    r_22 = q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2
    r_23 = 2*(q[2]*q[3] - q[0]*q[1])
    r_31 = 2*(q[1]*q[3] - q[0]*q[2])
    r_32 = 2*(q[2]*q[3] + q[0]*q[1])
    r_33 = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2

    return np.array([[r_11,r_12,r_13],[r_21,r_22,r_23],[r_31,r_32,r_33]])

def obtainComparableMatrix(acc_scale_matrix, acc_misal_matrix, gyro_scale_matrix, gyro_misal_matrix):
    # acc misal parameter taken from datasheet
    alpha_xz_6 = 0.01
    alpha_xy_6 = -0.02
    alpha_yx_6 = 0.01

    R_xz = quaternion(-alpha_xz_6,np.array([0,0,1]))
    R_xy = quaternion(-alpha_xy_6,np.array([0,1,0]))
    R_yx = quaternion(-alpha_yx_6,np.array([1,0,0]))

    comp_a_scale_matrix = np.linalg.inv(acc_scale_matrix)
    comp_a_misal_matrix = np.linalg.inv(np.matmul(np.matmul(np.matmul(R_xz,R_xy),R_yx),acc_misal_matrix))
    comp_g_scale_matrix = np.linalg.inv(gyro_scale_matrix)
    comp_g_misal_matrix = np.linalg.inv(np.matmul(np.matmul(np.matmul(R_xz,R_xy),R_yx),gyro_misal_matrix))

    return [comp_a_scale_matrix,comp_a_misal_matrix,comp_g_scale_matrix,comp_g_misal_matrix]
