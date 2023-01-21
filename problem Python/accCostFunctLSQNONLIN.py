# accCostFunctLSQNONLIN.py

def accCostFunctLSQNONLIN(E,a_hat):

    import numpy as np

    misalignmentMatrix = np.array([[1, -E[0], E[1]], [0, 1, -E[2]], [0, 0, 1]])
    scalingMtrix = np.diag([E[4], E[5], E[6]])

    a_bar = misalignmentMatrix*scalingMtrix*(a_hat) - (np.diag([E[7],E[8],E[9]])*np.ones([3,a_hat.shape[1]]))

    # Magnitude taken from tables
    magnitude = 9.81744

    residuals = np.zeros([a_hat.shape[1],1])

    for i in range (a_hat.shape[1]):
        residuals[i,0] = magnitude**2 - (a_bar[0,i]**2 + a_bar[1,i]**2 + a_bar[2,i]**2)

    res_vector = residuals
