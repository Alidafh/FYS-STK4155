#!/usr/bin/python
import numpy as np
import pandas as pd
from imageio import imread
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import functions as func
import plotting as plot
import sys


###############################################################################
def part_a(degree):
    print ("------------------------------------------------------")
    print ("                      PART A                          ")
    print ("------------------------------------------------------")

    plot.plot_franke("Illustration of the Franke Function", "franke_illustration")
    x, y, z = func.GenerateData(100, 0.1, "debug")
    X = func.PolyDesignMatrix(x, y, degree)

    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.33)
    X_train_scl, X_test_scl = func.scale_X(X_train, X_test)

    print("Fitting with OLS:")
    beta, conf_beta = func.OLS_SVD(z_train, X_train_scl, conf=True)

    z_train_fit = X_train_scl @ beta
    z_test_pred = X_test_scl @ beta

    R2_train, MSE_train, var_train, bias_train = func.metrics(z_train, z_train_fit)
    R2_test, MSE_test, var_test, bias_test = func.metrics(z_test, z_test_pred)
    #print ("----------------------")
    print ("    Deg : {}".format(degree))
    print ("    RS2 : {:.3f}".format(R2_train))
    print ("    MSE : {:.3f}".format(MSE_train))
    print ("    Beta:", np.array_str(beta.ravel(), precision=2, suppress_small=True))
    print ("    Conf:", np.array_str(conf_beta, precision=2, suppress_small=True))
    #print ("----------------------")
    print("")

part_a(2)

###############################################################################

print ("------------------------------------------------------")
print ("                      PART B                          ")
print ("------------------------------------------------------")

def part_b_noresample(d):
    print("Preforming OLS-regression using polynomials up to {} degrees, no resample\n".format(d))
    x, y, z = func.GenerateData(100, 0.01, "debug")
    #d = 20  # maximum number of polynomial degrees
    mse_test = np.zeros(d)
    mse_train = np.zeros(d)
    degrees = np.arange(1, d+1)

    for i in range(d):
        """ Loop over degrees"""
        X = func.PolyDesignMatrix(x, y, degrees[i])

        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.33)
        X_train_scl, X_test_scl = func.scale_X(X_train, X_test) # Scale data

        beta = func.OLS_SVD(z_train, X_train_scl)

        z_train_fit = X_train_scl @ beta
        z_test_pred = X_test_scl @ beta

        R2_train, MSE_train, var_train, bias_train = func.metrics(z_train, z_train_fit)
        R2_test, MSE_test, var_test, bias_test = func.metrics(z_test, z_test_pred)
        mse_train[i] = MSE_train
        mse_test[i] = MSE_test

    plot.plot_MSE(degrees, mse_test, mse_train, "OLS", "degrees_{}".format(d))
    print("-------------------------------------\n")


part_b_noresample(20)
###############################################################################

from sklearn.utils import resample

d = 5 # maximum number of polynomial degrees
n_bootstraps = 100   # number of bootsraps
print("Preforming OLS-regression using polynomials up to {:.0f} degrees with n_bootstrap={:.0f}\n".format(d, n_bootstraps))

x, y, z = func.GenerateData(100, 0.01, "debug")

# Initialize arrays of shape (degrees, )
bias = np.zeros(d)
variance = np.zeros(d)
mse = np.zeros(d)
error = np.zeros(d)
r2_score = np.zeros(d)

degrees = np.arange(1, d+1) # array of degrees

for i in range(d):
    """ Loop over degrees"""
    X = func.PolyDesignMatrix(x, y, degrees[i])

    # Split and scale data
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.33)
    X_train_scl, X_test_scl = func.scale_X(X_train, X_test)
    #z_test_pred = np.empty((z_test.shape[0], n_bootstraps))
    z_test_pred = np.empty((len(z_test), n_bootstraps))
    for j in range(n_bootstraps):
        """ Loop over bootstraps"""
        tmp_X_train, tmp_z_train = resample(X_train, z_train)
        tmp_beta = func.OLS_SVD(tmp_z_train, tmp_X_train)
        z_test_pred[:,j] = X_test_scl @ tmp_beta.ravel()

    # Calculate the stuff
    r2_score[i], mse[i], variance[i], bias[i] = func.metrics(z_test, z_test_pred)

plot.bias_variance(degrees, mse, variance, bias, rType = "OLS", c = "degrees_{}".format(d))
print("-------------------------------------")





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
