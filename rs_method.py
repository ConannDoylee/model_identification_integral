import numpy as np
from scipy.optimize import leastsq

## data 第一列为真值，后面所有列为特征
## initialTheta 估算的权值初值
## featureNum 特征的个数
def RLS_Fun(data, initialTheta, featureNum):
    theta_list = [[] for i in np.arange(featureNum)]
    Theta = initialTheta
    P = 10 ** 6 * np.eye(featureNum)
    lamda = 1
    for i in range(len(data)):
        featureMatrix = data[i][1:]
        featureMatrix = featureMatrix.reshape(featureMatrix.shape[0], 1)
        y_real = data[i][0]
        K = np.dot(P, featureMatrix) / (lamda + np.dot(np.dot(featureMatrix.T, P), featureMatrix))
        Theta = Theta + np.dot(K, (y_real - np.dot(featureMatrix.T, Theta)))
        P = np.dot((np.eye(featureNum) - np.dot(K, featureMatrix.T)), P)
        for i in np.arange(featureNum):
            theta_list[i].append(Theta[i][0])
    return Theta,theta_list