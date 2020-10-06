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
from array import array

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

    R2_train, MSE_train, var_train, bias_train = func.metrics(z_train, z_train_fit, test=True)
    R2_test, MSE_test, var_test, bias_test = func.metrics(z_test, z_test_pred,test=True)
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

#part_a(x,y,z,3)

###############################################################################
def part_b_noresample(x, y, z, d=5):
    print ("------------------------------------------------------")
    print ("                      PART B                          ")
    print ("                   no resampling                      ")
    print ("------------------------------------------------------")

    print("Preforming OLS-regression using polynomials up to {} degrees\n".format(d))
    degrees = np.arange(1, d+1)

    metrics_test = np.zeros((4, d))
    metrics_train = np.zeros((4, d))

    for i in range(d):
        """ Loop over degrees"""
        X = func.PolyDesignMatrix(x, y, degrees[i])

        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.33)
        X_train_scl, X_test_scl = func.scale_X(X_train, X_test)

        beta = func.OLS(z_train, X_train_scl)

        z_train_fit = X_train_scl @ beta
        z_test_pred = X_test_scl @ beta

        metrics_train[:,i] = func.metrics(z_train, z_train_fit, test=True)
        metrics_test[:,i] = func.metrics(z_test, z_test_pred, test=True)

    info = "n{:.0f}_d{:.0f}_noresample".format(len(z), d)

    plot.all_metrics_test_train(degrees, metrics_test, metrics_train, x_type="degrees", reg_type="OLS", other="w/o resample", info=info)

#part_b_noresample(x,y,z,d=10)

###############################################################################

def part_b_bootstrap(x, y, z, d=5, n_bootstraps=100, RegType="OLS"):
    print ("------------------------------------------------------")
    print ("                      PART B                          ")
    print ("                    resampling                        ")
    print ("------------------------------------------------------")

    print("Preforming OLS-regression using polynomials up to {:.0f} degrees with n_bootstrap={:.0f}\n".format(d, n_bootstraps))

    metrics_test = np.zeros((4, d))
    metrics_train = np.zeros((4, d))

    degrees = np.arange(1, d+1) # array of degrees

    for i in range(d):
        z_train, z_test, z_fit, z_pred = func.Bootstrap(x, y, z, degrees[i], n_bootstraps,"OLS", lamb=0)
        metrics_test[:,i] = func.metrics(z_test, z_pred, test=True)
        metrics_train[:,i] = func.metrics(z_train, z_fit, test=True)

    # Plotting
    info = "n{:.0f}_d{:.0f}_bs{:.0f}".format(len(z), d, n_bootstraps)
    plot.all_metrics_test_train(degrees, metrics_test, metrics_train, x_type="degrees", reg_type="OLS", other="Bootstrap", info=info)
    plot.bias_variance(degrees, metrics_test[1], metrics_test[2], metrics_test[3], "degrees","OLS", info, log=True)

#part_b_bootstrap(x, y, z, d=10, n_bootstraps=100, RegType="OLS")

###############################################################################

def part_b_datavariation(min, max, steps, d=5, n_bootstraps=100, RegType="OLS", lamb=0):
    print ("------------------------------------------------------")
    print ("                      PART B                          ")
    print ("                   data-variation                     ")
    print ("------------------------------------------------------")

    ndata = np.arange(min, max, steps)
    n = len(ndata)

    metrics_test = np.zeros((4, n))
    metrics_train = np.zeros((4, n))
    metrics_test_bs = np.zeros((4, n))
    metrics_train_bs = np.zeros((4, n))

    for i in range(n):
        x_tmp, y_tmp, z_tmp = func.GenerateData(ndata[i], 0.01, "debug", False)

        # Without resampling
        X_tmp = func.PolyDesignMatrix(x_tmp, y_tmp, d)
        X_train, X_test, z_train, z_test = train_test_split(X_tmp, z_tmp, test_size=0.33)
        X_train_scl, X_test_scl = func.scale_X(X_train, X_test)
        beta = func.OLS(z_train, X_train_scl)
        z_fit = X_train_scl @ beta      # (len(z_fit), 1)
        z_pred = X_test_scl @ beta      # (len(z_pred), 1)

        metrics_test[:,i] = func.metrics(z_test, z_pred, test=True)
        metrics_train[:,i] = func.metrics(z_train, z_fit, test=True)

        # With resampling
        z_train_bs, z_test_bs, z_fit_bs, z_pred_bs = func.Bootstrap(x_tmp, y_tmp, z_tmp, d, n_bootstraps, RegType, lamb)

        metrics_test_tmp = np.zeros((z_pred_bs.shape[1], 4))
        metrics_train_tmp = np.zeros((z_fit_bs.shape[1], 4))

        for j in range(z_fit_bs.shape[1]):
            z_fit_bs_j = z_fit_bs[:,j].reshape(-1,1)
            z_pred_bs_j = z_pred_bs[:,j].reshape(-1,1)
            metrics_train_tmp[j,:] = func.metrics(z_train_bs, z_fit_bs_j, test=True)
            metrics_test_tmp[j,:] = func.metrics(z_test_bs, z_pred_bs_j, test=True)

        metrics_train_bs[:,i] = np.mean(metrics_train_tmp, axis=0, keepdims=True)
        metrics_test_bs[:,i] = np.mean(metrics_test_tmp, axis=0, keepdims=True)


    info1 = "min{:.0f}_max{:.0f}_d{:.0f}_noresample".format(min, max, d, n_bootstraps)
    info2 = "min{:.0f}_max{:.0f}_d{:.0f}_bs{:.0f}_bootstrap".format(min, max, d, n_bootstraps)

    plot.all_metrics_test_train(ndata, metrics_test, metrics_train, x_type="data", reg_type=RegType, other="w/o resampling", info=info1)
    plot.all_metrics_test_train(ndata, metrics_test_bs, metrics_train_bs, x_type="data", reg_type=RegType, other="Bootstrap", info=info2)

    plot.bias_variance(ndata, metrics_test[1], metrics_test[2], metrics_test[3], "data", RegType, info1, log=True)
    plot.bias_variance(ndata, metrics_test_bs[1], metrics_test_bs[2], metrics_test_bs[3], "data", RegType, info2, log=True)

    
part_b_datavariation(min = 100, max=500, steps=50, d=1, n_bootstraps=100, RegType="OLS")
#part_b_datavariation(min = 100, max=500, steps=50, d=3, n_bootstraps=100, RegType="OLS")
#part_b_datavariation(min = 100, max=500, steps=50, d=7, n_bootstraps=100, RegType="OLS")
#part_b_datavariation(min = 100, max=500, steps=50, d=9, n_bootstraps=100, RegType="OLS")

part_b_datavariation(min = 100, max=500, steps=50, d=3, n_bootstraps=100, RegType="RIDGE", lamb=0.01)

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

#part_c_kFold(x,y,z, d=10, k=5, shuffle=True)

###############################################################################
def part_d_a(x, y, z, lamb, degree=5):
    print ("------------------------------------------------------")
    print ("                      PART D                          ")
    print ("------------------------------------------------------")

    X = func.PolyDesignMatrix(x, y, degree)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.33)
    X_train_scl, X_test_scl = func.scale_X(X_train, X_test)

    print("Fitting with Ridge:")
    beta, var_beta = func.Ridge(z_train, X_train_scl, lamb, var=True)
    conf_beta = 1.96*np.sqrt(var_beta)  # 95% confidence

    z_train_fit = X_train_scl @ beta
    z_test_pred = X_test_scl @ beta

    R2_train, MSE_train, var_train, bias_train = func.metrics(z_train, z_train_fit)
    R2_test, MSE_test, var_test, bias_test = func.metrics(z_test, z_test_pred)
    #print ("----------------------")
    print ("    lamb: {}".format(lamb))
    print ("    Deg : {}".format(degree))
    print ("    RS2 : {:.3f} (train: {:.3f})".format(R2_test, R2_train))
    print ("    MSE : {:.3f} (train: {:.3f})".format(MSE_test, MSE_train))
    print ("    Var : {:.3f} (train: {:.3f})".format(var_test, var_train))
    print ("    Bias: {:.3f} (train: {:.3f})".format(bias_test, bias_train))
    print ("    Beta:", np.array_str(beta.ravel(), precision=2, suppress_small=True))
    print ("    Conf:", np.array_str(conf_beta.ravel(), precision=2, suppress_small=True))
    print ("")
    #print ("----------------------")

    plot.RIDGE_beta_conf(beta, conf_beta, degree, lamb, len(z))
    return beta


###############################################################################

def params_vs_lambda(x, y, z, d, nlamb):
    """ not done"""
    bl0 = part_d_a(x, y, z, 0, d)   # just to get the size
    p = len(bl0)
    lambdas = np.logspace(-3, 0, nlamb)
    betas = np.zeros((len(bl0), len(lambdas)))

    metrics_test = np.zeros((4, nlamb))
    metrics_train = np.zeros((4, nlamb))

    for i in range(nlamb):
        X = func.PolyDesignMatrix(x, y, d)
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.33)
        X_train_scl, X_test_scl = func.scale_X(X_train, X_test)

        beta= func.Ridge(z_train, X_train_scl, lambdas[i])

        z_train_fit = X_train_scl @ beta
        z_test_pred = X_test_scl @ beta
        metrics_train[:,i] = func.metrics(z_train, z_train_fit)
        metrics_test[:,i] = func.metrics(z_test, z_test_pred)
        betas[:,i] = beta.ravel()

    plt.plot(lambdas, betas.T)
    plt.show()

#params_vs_lambda(x,y,z,3,5)
###############################################################################

def part_d_bias_variance(x, y, z, d, lamb):
    print ("------------------------------------------------------")
    print ("                      PART D                          ")
    print ("                   bias-variance                      ")
    print ("------------------------------------------------------")

    print("Preforming Ridge-regression using polynomials up to {} degrees and lambda {}\n".format(d, lamb))

    degrees = np.arange(1, d+1)
    metrics_test = np.zeros((4, d))
    metrics_train = np.zeros((4, d))

    for i in range(d):
        """ Loop over degrees"""
        X = func.PolyDesignMatrix(x, y, degrees[i])
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.33)
        X_train_scl, X_test_scl = func.scale_X(X_train, X_test)
        beta = func.Ridge(z_train, X_train_scl, lamb)
        z_train_fit = X_train_scl @ beta
        z_test_pred = X_test_scl @ beta

        metrics_train[:,i] = func.metrics(z_train, z_train_fit, test=True)
        metrics_test[:,i] = func.metrics(z_test, z_test_pred, test=True)

        # Bootstrap:
        z_train_bs, z_test_bs, z_fit_bs, z_pred_bs = func.Bootstrap(x_tmp, y_tmp, z_tmp, d, n_bootstraps, "OLS", lamb=0)


    info = "ndata{:.0f}_d{:.0f}".format(len(z), d)
    plot.RIDGE_test_train(degrees, mse_test, mse_train, lamb, "MSE", info, log=True)

#part_d_noresamle(x, y, z, 10, 0.0001)

###############################################################################
"""
def simple_ridge(x, y, z, lamb, degree=5):
    X = func.PolyDesignMatrix(x, y, degree)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.33)
    X_train_scl, X_test_scl = func.scale_X(X_train, X_test)

    beta, var_beta = func.Ridge(z_train, X_train_scl, lamb, var=True)
    conf_beta = 1.96*np.sqrt(var_beta)  # 95% confidence

    z_train_fit = X_train_scl @ beta
    z_test_pred = X_test_scl @ beta

    R2_train, MSE_train, var_train, bias_train = func.metrics(z_train, z_train_fit)
    R2_test, MSE_test, var_test, bias_test = func.metrics(z_test, z_test_pred)

    return MSE_test, var_test, bias_test

nlambdas = 10
lambdas = np.linspace(0, 20, nlambdas)
mse_ridge = np.zeros(nlambdas)
var_ridge = np.zeros(nlambdas)
bias_ridge = np.zeros(nlambdas)
for i in range(nlambdas):
    mse_ridge[i], var_ridge[i], bias_ridge[i] = simple_ridge(x,y,z, lambdas[i], degree=3)


plt.plot(lambdas, mse_ridge, "o")
plt.plot(lambdas, var_ridge, "o")
plt.plot(lambdas, bias_ridge, "o")
plt.show()
"""
###############################################################################



############################# DO NOT ERASE ####################################
########################### (Without asking) ##################################

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
