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
# Simple OLS
def part_a(x, y, z, degree=5, bplot=False):
    print ("------------------------------------------------------")
    print ("                      PART A                          ")
    print ("------------------------------------------------------")

    #plot.plot_franke("Illustration of the Franke Function", "franke_illustration", 0.1)

    X = func.PolyDesignMatrix(x, y, degree)

    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.33)

    X_train_scl, X_test_scl = func.scale_X(X_train, X_test)

    print("Fitting with OLS:")
    beta, var_beta = func.OLS(z_train, X_train_scl, var=True)
    conf_beta = 1.96*np.sqrt(var_beta)  # 95% confidence

    z_train_fit = X_train_scl @ beta
    z_test_pred = X_test_scl @ beta

    R2_train, MSE_train, tmp, tmp= func.metrics(z_train, z_train_fit, test=True)
    R2_test, MSE_test, tmp, tmp = func.metrics(z_test, z_test_pred,test=True)
    print ("    Deg : {}".format(degree))
    print ("    RS2 : {:.3f} (train: {:.3f})".format(R2_test, R2_train))
    print ("    MSE : {:.3f} (train: {:.3f})".format(MSE_test, MSE_train))
    print ("    Beta:", np.array_str(beta.ravel(), precision=2, suppress_small=True))
    print ("    Conf:", np.array_str(conf_beta.ravel(), precision=2, suppress_small=True))
    print ("")

    if bplot==True: plot.OLS_beta_conf(beta, conf_beta, degree, len(z))

part_a(x,y,z,3, bplot=True)

###############################################################################
# Vary the degrees and look at bias-variance and r2 and test-train
def part_b_noresample(x, y, z, d=5, bplot=False):
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

    if bplot==True:
        plot.OLS_test_train(degrees, metrics_test[1], metrics_train[1], err_type ="MSE", info=info, log=True)

part_b_noresample(x, y, z, d=10, bplot=True)

###############################################################################
# Bootstrap for OLS
def part_b_bootstrap(x, y, z, d=5, n_bootstraps=100, RegType="OLS",lamb=0, bplot=False):
    if bplot==True:
        print ("------------------------------------------------------")
        print ("                      PART B                          ")
        print ("                    resampling                        ")
        print ("------------------------------------------------------")
        print("Preforming OLS-regression using polynomials up to {:.0f} degrees with n_bootstrap={:.0f}\n".format(d, n_bootstraps))

    metrics_test = np.zeros((4, d))
    metrics_train = np.zeros((4, d))

    degrees = np.arange(1, d+1) # array of degrees

    for i in range(d):
        z_train, z_test, z_fit, z_pred = func.Bootstrap(x, y, z, degrees[i], n_bootstraps,RegType, lamb)
        metrics_test_tmp = np.zeros((z_pred.shape[1], 4))
        metrics_train_tmp = np.zeros((z_fit.shape[1], 4))
        """
        for j in range(z_fit.shape[1]):
            z_fit_bs_j = z_fit[:,j].reshape(-1,1)
            z_pred_bs_j = z_pred[:,j].reshape(-1,1)
            metrics_train_tmp[j,:] = func.metrics(z_train, z_fit_bs_j, test=True)
            metrics_test_tmp[j,:] = func.metrics(z_test, z_pred_bs_j, test=True)

        metrics_train[:,i] = np.mean(metrics_train_tmp, axis=0, keepdims=True)
        metrics_test[:,i] = np.mean(metrics_test_tmp, axis=0, keepdims=True)
        """

        metrics_test[:,i] = func.metrics(z_test, z_pred, test=True)
        metrics_train[:,i] = func.metrics(z_train, z_fit, test=True)

    # Plotting
    info = "n{:.0f}_d{:.0f}_bs{:.0f}".format(len(z), d, n_bootstraps)
    #if bplot==True: plot.all_metrics_test_train(degrees, metrics_test, metrics_train, x_type="degrees", reg_type = RegType, other="Bootstrap", info=info)
    if bplot==True: plot.bias_variance(degrees, metrics_test[1], metrics_test[2], metrics_test[3], "degrees",RegType, info, log=True)

    return degrees, metrics_test[1]

part_b_bootstrap(x, y, z, d=10, n_bootstraps=100, RegType="OLS", bplot=True)

###############################################################################
# Vary datasize

def part_b_datavariation(min, max, steps, d=5, n_bootstraps=100, RegType="OLS", lamb=0, bplot=False):
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

    if bplot==True:
        plot.all_metrics_test_train(ndata, metrics_test, metrics_train, x_type="data", reg_type=RegType, other="w/o resampling", info=info1)
        plot.all_metrics_test_train(ndata, metrics_test_bs, metrics_train_bs, x_type="data", reg_type=RegType, other="Bootstrap", info=info2)

        plot.bias_variance(ndata, metrics_test[1], metrics_test[2], metrics_test[3], "data", RegType, info1, log=True)
        plot.bias_variance(ndata, metrics_test_bs[1], metrics_test_bs[2], metrics_test_bs[3], "data", RegType, info2, log=True)

part_b_datavariation(min = 100, max=500, steps=50, d=1, n_bootstraps=100, RegType="OLS")
#part_b_datavariation(min = 100, max=500, steps=50, d=3, n_bootstraps=100, RegType="OLS")
#part_b_datavariation(min = 100, max=500, steps=50, d=7, n_bootstraps=100, RegType="OLS")
#part_b_datavariation(min = 100, max=500, steps=50, d=9, n_bootstraps=100, RegType="OLS")

###############################################################################
# K-fold resampling for OLS
def part_c_kFold(x, y, z, d=5, k=5, shuffle = False, RegType="OLS", lamb=0, bplot=False):
    """
    Uses folding to split the data
    --------------------------------
    Input
        x,y,z: the data
        d: maximum number of degrees
        k: number of folds
        shuffle: if the data should be randomized, default is False
    --------------------------------
    TO DO:
    """
    if bplot==True:
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

            if RegType == "OLS": beta = func.OLS(z_train, X_train_scl)
            if RegType == "RIDGE": beta = func.Ridge(z_train, X_train_scl, lamb)

            z_fit = X_train_scl @ beta
            z_pred = X_test_scl @ beta

            rs2_kFold[deg_i,fold_i], mse_kFold[deg_i,fold_i], var_kFold[deg_i,fold_i], bias_kFold[deg_i,fold_i] = func.metrics(z_test, z_pred, test=True)

            fold_i +=1
        deg_i +=1
    if bplot==True:
        plot.allfolds(degrees, mse_kFold, k, len(z), rType=RegType, varN="MSE", log=True, lamb=lamb)
        plot.allfolds(degrees, rs2_kFold, k, len(z), rType=RegType, varN="R2", log=False, lamb=lamb)
        plot.allfolds(degrees, var_kFold, k, len(z), rType=RegType, varN="Variance",log=True, lamb=lamb)
        plot.allfolds(degrees, bias_kFold, k, len(z), rType=RegType, varN="Bias",log=True, lamb=lamb)

    est_rs2_kFold = np.mean(rs2_kFold, axis = 1)
    est_mse_kFold = np.mean(mse_kFold, axis = 1)
    est_var_kFold = np.mean(var_kFold, axis = 1)
    est_bias_kFold = np.mean(bias_kFold, axis = 1)

    info = "data{:.0f}_degree{:.0f}_kFold{:.0f}".format(len(z), d, k)
    if bplot==True:
        plot.bias_variance(degrees, est_mse_kFold, est_var_kFold, est_bias_kFold, "degrees", RegType, info, log=True)
        plot.OLS_metric(degrees, est_rs2_kFold, "R2-score", info, log=False)

    return degrees, est_mse_kFold

part_c_kFold(x, y, z, d=10, k=5, shuffle=True, RegType="OLS", bplot=True)
################################################################################
#Comparison for OLS

deg1, mse_kFold = part_c_kFold(x, y, z, d=10, k=5, shuffle=True, RegType="OLS", bplot=False)
deg2, mse_boots = part_b_bootstrap(x, y, z, d=10, n_bootstraps=100, RegType="OLS", bplot=False)

plot.compare_MSE(deg1, mse_kFold, mse_boots, rType = "OLS", lamb=0, info="test", log=False)

###############################################################################
# Simple ridge regression

def part_d_a(x, y, z, lamb, degree=5, bplot=False):
    if bplot==True:
        print ("------------------------------------------------------")
        print ("                      PART D                          ")
        print ("------------------------------------------------------")
        print("Fitting with Ridge:")
    X = func.PolyDesignMatrix(x, y, degree)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.33)
    X_train_scl, X_test_scl = func.scale_X(X_train, X_test)


    beta, var_beta = func.Ridge(z_train, X_train_scl, lamb, var=True)
    conf_beta = 1.96*np.sqrt(var_beta)  # 95% confidence

    z_train_fit = X_train_scl @ beta
    z_test_pred = X_test_scl @ beta

    R2_train, MSE_train, var_train, bias_train = func.metrics(z_train, z_train_fit)
    R2_test, MSE_test, var_test, bias_test = func.metrics(z_test, z_test_pred)
    if bplot==True:
        print ("    lamb: {}".format(lamb))
        print ("    Deg : {}".format(degree))
        print ("    RS2 : {:.3f} (train: {:.3f})".format(R2_test, R2_train))
        print ("    MSE : {:.3f} (train: {:.3f})".format(MSE_test, MSE_train))
        print ("    Var : {:.3f} (train: {:.3f})".format(var_test, var_train))
        print ("    Bias: {:.3f} (train: {:.3f})".format(bias_test, bias_train))
        print ("    Beta:", np.array_str(beta.ravel(), precision=2, suppress_small=True))
        print ("    Conf:", np.array_str(conf_beta.ravel(), precision=2, suppress_small=True))
        print ("")

    if bplot==True: plot.RIDGE_beta_conf(beta, conf_beta, degree, lamb, len(z))
    return beta

part_d_a(x,y,z,0.1, 3, bplot=True)

###############################################################################
# Comparison of MSE bootstrap vs kFold for Ridge

deg1, mse_kFold = part_c_kFold(x, y, z, d=10, k=5, shuffle=True, RegType="RIDGE",lamb=0.1, bplot=False)
deg2, mse_boots = part_b_bootstrap(x, y, z, d=10, n_bootstraps=100, RegType="RIDGE",lamb=0.1, bplot=False)

plot.compare_MSE(deg1, mse_kFold, mse_boots, rType = "RIDGE", lamb=0.1, info="test", log=False)

###############################################################################
# Vary lambda and calculate bias-variance with and without kFold

def part_d_varylambda(x, y, z, d, nlamb, k, shuffle=False, bplot=False):
    """ not done"""

    bl0 = part_d_a(x, y, z, 0, d, bplot=False)   # just to get the size
    p = len(bl0)
    lambdas = np.logspace(-3, 0, nlamb)

    betas = np.zeros((len(bl0), len(lambdas)))

    metrics_test = np.zeros((4, nlamb))
    metrics_train = np.zeros((4, nlamb))

    mse_kFold = np.zeros((nlamb,k))        # arrays of statistics with d rows and
    bias_kFold = np.zeros((nlamb,k))       # k columns. Each row corresponds to a
    rs2_kFold = np.zeros((nlamb,k))        # degree and each colunm corresponds to
    var_kFold = np.zeros((nlamb,k))        # to the fold number

    lamb_i = 0
    for i in range(nlamb):
        # No resampling
        X = func.PolyDesignMatrix(x, y, d)
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.33)
        X_train_scl, X_test_scl = func.scale_X(X_train, X_test)

        beta = func.Ridge(z_train, X_train_scl, lambdas[i])
        betas[:,i] = beta.ravel()
        z_train_fit = X_train_scl @ beta
        z_test_pred = X_test_scl @ beta
        metrics_train[:,i] = func.metrics(z_train, z_train_fit)
        metrics_test[:,i] = func.metrics(z_test, z_test_pred)

        # k-fold:
        np.random.seed(42)
        if shuffle == True: np.random.shuffle(X) # Shuffle the rows
        fold_i = 0
        for j in range(1, k+1):
            #loop over folds and calculate the fitted and predicted z values
            train_index, test_index = tools.foldIndex(x, j, k)
            X_train = X[train_index]
            z_train = z[train_index]

            X_test = X[test_index]
            z_test = z[test_index]

            X_train_scl, X_test_scl = func.scale_X(X_train, X_test)

            beta = func.Ridge(z_train, X_train_scl, lambdas[i])

            z_fit = X_train_scl @ beta
            z_pred = X_test_scl @ beta

            rs2_kFold[lamb_i,fold_i], mse_kFold[lamb_i,fold_i], var_kFold[lamb_i,fold_i], bias_kFold[lamb_i,fold_i] = func.metrics(z_test, z_pred, test=True)
            fold_i +=1


        lamb_i +=1

    est_rs2_kFold = np.mean(rs2_kFold, axis = 1)
    est_mse_kFold = np.mean(mse_kFold, axis = 1)
    est_var_kFold = np.mean(var_kFold, axis = 1)
    est_bias_kFold = np.mean(bias_kFold, axis = 1)

    if bplot==True:
        info1 = "ndata{:.0f}_d{:.0f}_noresample".format(len(z), d)
        info2 = "ndata{:.0f}_d{:.0f}_kfold{:.0f}".format(len(z), d,k)
        plot.bias_variance(lambdas, metrics_test[1], metrics_test[2], metrics_test[3], x_type="lambdas", RegType ="RIDGE", info=info1, log=False)
        plot.bias_variance(lambdas, est_mse_kFold, est_var_kFold, est_bias_kFold, "lambdas", RegType="RIDGE", info=info2, log=False)

part_d_varylambda(x, y, z, d=3, nlamb=5, k=5, shuffle=True, bplot=True)

###############################################################################
# vary data for ridge

part_b_datavariation(min = 100, max=500, steps=50, d=3, n_bootstraps=100, RegType="RIDGE", lamb=0.01, bplot=True)
###############################################################################

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
