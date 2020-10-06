#!/usr/bin/python
import numpy as np
import pandas as pd
from imageio import imread
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import functions as func
import plotting as plot
import tools as tools
import sys

#x, y, z = func.GenerateData(100, 0, "debug")
x, y, z = func.GenerateData(100, 0.01, "debug")


###############################################################################
def part_a(x, y, z, degree=5):
    print ("------------------------------------------------------")
    print ("                      PART A                          ")
    print ("------------------------------------------------------")

    plot.plot_franke("Illustration of the Franke Function", "franke_illustration", 0.1)

    X = func.PolyDesignMatrix(x, y, degree)

    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.33)

    X_train_scl, X_test_scl = func.scale_X(X_train, X_test)

    print("Fitting with OLS:")
    beta, var_beta = func.OLS(z_train, X_train_scl, var=True)
    conf_beta = 1.96*np.sqrt(var_beta)  # 95% confidence

    z_train_fit = X_train_scl @ beta
    z_test_pred = X_test_scl @ beta

    R2_train, MSE_train, var_train, bias_train = func.metrics(z_train, z_train_fit)
    R2_test, MSE_test, var_test, bias_test = func.metrics(z_test, z_test_pred)
    #print ("----------------------")
    print ("    Deg : {}".format(degree))
    print ("    RS2 : {:.3f} (train: {:.3f})".format(R2_test, R2_train))
    print ("    MSE : {:.3f} (train: {:.3f})".format(MSE_test, MSE_train))
    print ("    Var : {:.3f} (train: {:.3f})".format(var_test, var_train))
    print ("    Bias: {:.3f} (train: {:.3f})".format(bias_test, bias_train))
    print ("    Beta:", np.array_str(beta.ravel(), precision=2, suppress_small=True))
    print ("    Conf:", np.array_str(conf_beta.ravel(), precision=2, suppress_small=True))
    print ("")
    #print ("----------------------")

    plot.OLS_beta_conf(beta, conf_beta, degree, len(z))

part_a(x,y,z,3)

###############################################################################


def part_b_noresample(x, y, z, d=5):
    print ("------------------------------------------------------")
    print ("                      PART B                          ")
    print ("                   no resampling                      ")
    print ("------------------------------------------------------")

    print("Performing OLS-regression using polynomials up to {} degrees\n".format(d))
    mse_test = np.zeros(d)
    mse_train = np.zeros(d)
    degrees = np.arange(1, d+1)

    for i in range(d):
        """ Loop over degrees"""
        X = func.PolyDesignMatrix(x, y, degrees[i])

        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.33)
        X_train_scl, X_test_scl = func.scale_X(X_train, X_test) # Scale data

        beta = func.OLS(z_train, X_train_scl)

        z_train_fit = X_train_scl @ beta
        z_test_pred = X_test_scl @ beta

        R2_train, MSE_train, var_train, bias_train = func.metrics(z_train, z_train_fit)
        R2_test, MSE_test, var_test, bias_test = func.metrics(z_test, z_test_pred)

        mse_train[i] = MSE_train
        mse_test[i] = MSE_test

    info = "ndata{:.0f}_d{:.0f}".format(len(z), d)
    plot.OLS_test_train(degrees, mse_test, mse_train, err_type ="MSE", info="", log=True)

part_b_noresample(x,y,z,d=10)

###############################################################################


def part_b_bootstrap(x, y, z, d=5, n_bootstraps=100):
    print ("------------------------------------------------------")
    print ("                      PART B                          ")
    print ("                    resampling                        ")
    print ("------------------------------------------------------")

    print("Performing OLS-regression using polynomials up to {:.0f} degrees with n_bootstrap={:.0f}\n".format(d, n_bootstraps))

    # Initialize arrays of shape (degrees, )
    bias = np.zeros(d)
    var = np.zeros(d)
    mse = np.zeros(d)
    r2_score = np.zeros(d)
    degrees = np.arange(1, d+1) # array of degrees

    for i in range(d):
        """ Loop over degrees"""
        X = func.PolyDesignMatrix(x, y, degrees[i])

        # Split and scale data
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
        X_train_scl, X_test_scl = func.scale_X(X_train, X_test)

        z_pred = np.empty((z_test.shape[0], n_bootstraps))
        for j in range(n_bootstraps):
            """ Loop over bootstraps"""
            tmp_X_train, tmp_z_train = resample(X_train, z_train)
            tmp_beta = func.OLS(tmp_z_train, tmp_X_train)
            z_pred[:,j] = X_test_scl @ tmp_beta.ravel()

        r2_score[i], mse[i], var[i], bias[i] = func.metrics(z_test, z_pred, test=True)

    # Plotting
    info = "data{:.0f}_degree{:.0f}_bootstrap{:.0f}".format(len(z), d, n_bootstraps)
    plot.OLS_bias_variance(degrees, mse, var, bias, info, log=True)

    #plot.OLS_metric(degrees, r2_score, "R2-score", info)

part_b_bootstrap(x, y, z, d=10, n_bootstraps=100)

###############################################################################

def part_c_kFold(x, y, z, d=5, k=5, shuffle = False):
    """
    Uses folding to split the data
    --------------------------------
    Input
        x,y,z: the data
        d: maximum number of degrees
        k: number of folds
        shuffle: if the data should be randomized, default is False
    --------------------------------
    TO DO: The calcualted MSE and Bias (calculated using metrics) is identical..
            must find out why..
    """
    print ("------------------------------------------------------")
    print ("                      PART C                          ")
    print ("                      k-fold                          ")
    print ("------------------------------------------------------")

    degrees = np.arange(1, d+1)

    mse_kFold = np.zeros((d,k))        # arrays of statistics with d rows and
    bias_kFold = np.zeros((d,k))       # k columns. Each row corresponds to a
    rs2_kFold = np.zeros((d,k))        # degree and each colunm corresponds to
    var_kFold = np.zeros((d,k))        # to the fold number

    deg_i = 0
    for j in range(d):
        """loop over degrees"""
        degree = degrees[j]
        X = func.PolyDesignMatrix(x, y, degree)
        np.random.seed(42)
        if shuffle == True: np.random.shuffle(X) # Shuffle the rows
        fold_i = 0
        for i in range(1, k+1):
            """loop over folds and calculate the fitted and predicted z values"""
            train_index, test_index = tools.foldIndex(x, i, k)
            X_train = X[train_index]
            z_train = z[train_index]

            X_test = X[test_index]
            z_test = z[test_index]

            X_train_scl, X_test_scl = func.scale_X(X_train, X_test)

            beta = func.OLS(z_train, X_train_scl)

            z_fit = X_train_scl @ beta
            z_pred = X_test_scl @ beta

            rs2_kFold[deg_i,fold_i], mse_kFold[deg_i,fold_i], var_kFold[deg_i,fold_i], bias_kFold[deg_i,fold_i]= func.metrics(z_test, z_pred, test=True)

            fold_i +=1
        deg_i +=1

    plot.OLS_allfolds(degrees, mse_kFold, k, len(z), rType="OLS", varN="MSE", log=True)
    plot.OLS_allfolds(degrees, rs2_kFold, k, len(z), rType="OLS", varN="R2")
    plot.OLS_allfolds(degrees, var_kFold, k, len(z), rType="OLS", varN="Variance",log=True)
    plot.OLS_allfolds(degrees, bias_kFold, k, len(z), rType="OLS", varN="Bias",log=True)

    # np.mean(matrix, axis=1) takes the mean of the numbers in each row
    #
    #   M = [[m11  m12 ... m1k]
    #        [m21  m22 ... m2k]
    #        [ .    .       . ]
    #        [md1  md2 ... mdk]]
    #
    # np.mean(M, axis=1) = 1/k * [sum_i(m1i)  sum_i(m2i) ... sum_i(mdi)]

    est_rs2_kFold = np.mean(rs2_kFold, axis = 1)
    est_mse_kFold = np.mean(mse_kFold, axis = 1)
    est_var_kFold = np.mean(var_kFold, axis = 1)
    est_bias_kFold = np.mean(bias_kFold, axis = 1)

    info = "data{:.0f}_degree{:.0f}_kFold{:.0f}".format(len(z), d, k)
    plot.OLS_bias_variance(degrees, est_mse_kFold, est_var_kFold, est_bias_kFold, info, log=True)
    plot.OLS_metric(degrees, est_rs2_kFold, "R2-score", info, log=False)


part_c_kFold(x,y,z, d=10, k=5, shuffle=True)
#part_c_kFold(x,y,z, d=10, k=5)


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
