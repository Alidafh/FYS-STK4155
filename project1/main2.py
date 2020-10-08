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

def R2(z_true, z_pred):
    if z_pred.shape != z_true.shape:
        z_pred = np.mean(z_pred, axis=1, keepdims=True)

    return 1 - (np.sum((z_true - z_pred)**2))/(np.sum((z_true - np.mean(z_true))**2))

def MSE(z_true, z_pred):
    return np.mean(np.mean((z_true - z_pred)**2, axis=1, keepdims=True))

def Variance(z_true, z_pred):
    return np.mean(np.var(z_pred, axis=1, keepdims=True))

def Bias(z_true, z_pred):
    return np.mean((z_true - np.mean(z_pred, axis=1, keepdims=True))**2)

###############################################################################

def regression(z, X, reg_type="OLS", lamb=0):
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.33)
    X_train_scl, X_test_scl = func.scale_X(X_train, X_test)

    if reg_type=="OLS": beta = func.OLS(z_train, X_train_scl)
    if reg_type =="RIDGE": beta = func.RIDGE(z_train, X_train_scl, lamb)

    z_fit = X_train_scl @ beta
    z_pred = X_test_scl @ beta

    return z_train, z_test, z_fit, z_pred

def get_beta(x, y, z, d, reg_type="OLS", lamb=0):

    X = func.PolyDesignMatrix(x, y, d)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.33)
    X_train_scl, X_test_scl = func.scale_X(X_train, X_test)

    if reg_type=="OLS": beta, var_beta = func.OLS(z_train, X_train_scl, var=True)
    if reg_type =="RIDGE": beta, var_beta = func.RIDGE(z_train, X_train_scl, lamb, var=True)

    conf_beta = 1.96*np.sqrt(var_beta)  # 95% confidence

    z_fit = X_train_scl @ beta
    z_pred = X_test_scl @ beta

    #rs2, mse = func.metrics(z_test,z_pred, test=True)[:2]
    #print("{:} parameters for degree {:.0f} (lambda={:.6f}) ".format(reg_type, d, lamb))
    #print("    R2:  {:.4f}".format(rs2))
    #print("    MSE: {:.4f}\n".format(mse))

    return beta, conf_beta
###############################################################################
x, y, z = func.GenerateData(100, 0.01, "debug")


# Loop over degrees and do regression
d_max = 10
degrees = np.arange(1, d_max+1)

# Initialise arrays
m_test = np.zeros((4, d_max))       # array to store [R2, mse, var, bias]
m_train = np.zeros((4, d_max))      # array to store [R2, mse, var, bias]

for i in range(d_max):
    X = func.PolyDesignMatrix(x, y, degrees[i])
    z_train, z_test, z_fit, z_pred = regression(z, X, "OLS")
    m_test[:,i] = func.metrics(z_test, z_pred)
    m_train[:,i] = func.metrics(z_train, z_fit)

# Recreate figure 2.11 in Hastie
info1 = "n{:.0f}_d{:.0f}".format(len(z), d_max)
plot.OLS_test_train(degrees, m_test[1], m_train[1], err_type ="MSE", info=info1, log=True)


# With bootstrapping
n_bootstraps = 100

m_test_bs, m_train_bs = func.OLS_bootstrap_degrees(x, y, z, degrees, n_bootstraps)

info_bs = info1+"_bs{}".format(n_bootstraps)
plot.OLS_bias_variance(degrees, m_test_bs[1], m_test_bs[2], m_test_bs[3], "degrees", info_bs, log=True)
plot.OLS_test_train(degrees, m_test_bs[1], m_train_bs[1], "MSE", info_bs, log=True)
plot.all_metrics_test_train(degrees, m_test_bs, m_train_bs, "degrees", "OLS", "Bootstrap", info_bs)

# find the regression parameters for the best polynomial degree
#min_mse = m_test[1].min()
#best_d_ols = m_test[1].argmin() +1  #python starts counting on zero

#beta, conf_beta = get_beta(x, y, z, best_d_ols, "OLS")
#plot.OLS_beta_conf(beta, conf_beta, best_d_ols, len(z))

quit()

min =100
max = 500
steps =50

ndata = np.arange(min, max, steps)
n = len(ndata)

deg = 2
m_test_bs = np.zeros((4, n))
m_train_bs = np.zeros((4, n))

for i in range(n):
    x_tmp, y_tmp, z_tmp = func.GenerateData(ndata[i], 0.01, "debug", pr=False)

    # First without resampling
    X_tmp = func.PolyDesignMatrix(x_tmp, y_tmp, deg)
    z_train, z_test, z_fit, z_pred = regression(z_tmp, X_tmp, "OLS")

    m_test[:,i] = func.metrics(z_test, z_pred, test=True)
    m_train[:,i] = func.metrics(z_train, z_fit, test=True)

    # With resampling
    z_train_bs, z_test_bs, z_fit_bs, z_pred_bs = func.Bootstrap(x_tmp, y_tmp, z_tmp, deg, n_bootstraps, "OLS")

    m_test_tmp = np.zeros((z_pred_bs.shape[1], 4))
    m_train_tmp = np.zeros((z_fit_bs.shape[1], 4))

    for j in range(z_fit_bs.shape[1]):
        z_fit_bs_j = z_fit_bs[:,j].reshape(-1,1)
        z_pred_bs_j = z_pred_bs[:,j].reshape(-1,1)
        m_train_tmp[j,:] = func.metrics(z_train_bs, z_fit_bs_j, test=True)
        m_test_tmp[j,:] = func.metrics(z_test_bs, z_pred_bs_j, test=True)

    m_train_bs[:,i] = np.mean(m_train_tmp, axis=0, keepdims=True)
    m_test_bs[:,i] = np.mean(m_test_tmp, axis=0, keepdims=True)

plt.figure()
plt.plot(ndata, m_test_bs[0])
plt.show()
