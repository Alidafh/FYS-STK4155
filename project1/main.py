#!/usr/bin/python
import numpy as np
import pandas as pd
from imageio import imread
import general_functions as func
import plotting as plot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

plot.plot_franke("Illustration of the Franke Function", "franke_func_illustration")
x, y, z = func.GenerateData(1000, 0.1, "debug")

d = 20

mse_test = np.zeros(d)
mse_train = np.zeros(d)
degrees = np.arange(1, d+1)

for i in range(d):
    X = func.PolyDesignMatrix(x, y, degrees[i])
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.33)

    # Scale data
    X_train_scl, X_test_scl = func.scale_X(X_train, X_test)

    beta, conf_inter = func.OLS(z_train, X_train_scl)

    z_train_fit = X_train_scl @ beta
    z_test_pred = X_test_scl @ beta

    R2_train, MSE_train, var_train, bias_train = func.metrics(z_train, z_train_fit)
    R2_test, MSE_test, var_test, bias_test = func.metrics(z_test, z_test_pred)
    mse_train[i] = MSE_train
    mse_test[i] = MSE_test


plot.plot_MSE(degrees, mse_test, mse_train, "OLS")
plt.show()


"""
plt.figure()
#plt.plot(degrees, mse_train, degrees, mse_test)
plt.plot(degrees, bias_test,  degrees, variance_test, degrees, mse_test)
plt.legend(["bias", "variance", "MSE"])
plt.xlabel("complexity")
plt.ylabel(" y-axis")
plt.show()
"""

"""
plt.figure()
#plt.plot(degrees, mse_train, degrees, mse_test)
plt.plot(degrees, bias_train, degrees, bias_test, degrees, variance_train, degrees, variance_test)
plt.legend(["bias train", "bias test", "var train", "var test"])
plt.xlabel("complexity")
plt.ylabel(" y-axis")
plt.show()

plt.figure()
plt.plot(degrees, mse_train, degrees, mse_test)
#plt.plot(degrees, bias_train, degrees, bias_test, degrees, variance_train, degrees, variance_test)
plt.legend(["train", "test"])
plt.xlabel("complexity")
plt.ylabel("MSE")
plt.show()
"""
############################# DO NOT ERASE ####################################
########################### (Without asking) ####################################

"""
scaler = StandardScaler()
scaler.fit(X_train[:,1:])
X_train_scaled = scaler.transform(X_train[:,1:])
#X_test_scaled = scaler.transform(X_test)

n = len(X_train_scaled[:,1])
ones = np.ones((n,1))

X_train_new = np.hstack((ones, X_train_scaled))

z_train_predict, beta, conf_inter_beta = func.OLS(z_train, X_train_new)
R2, MSE, var = func.metrics(z_train, z_train_predict)
print("---sklear--------")
print("R2: ", R2)
print("MSE: ", MSE)

"""

"""
terrain1 = imread("datafiles/SRTM_data_Norway_1.tif")

plt.figure()
plt.title("Terrain over Norway 1")
plt.imshow(terrain1, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
"""
