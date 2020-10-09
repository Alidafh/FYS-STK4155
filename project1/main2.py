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

###############################################################################

def GenDat2():
    x_ = np.arange(0, 1, 0.1)
    y_ = np.arange(0, 1, 0.1)

    x, y = np.meshgrid(x_,y_)

    noise = 0.01
    z = func.FrankeFunction(x,y) + np.random.normal(0, noise, len(x))

    x = x.ravel().reshape(-1,1)
    y = y.ravel().reshape(-1,1)
    z = z.ravel().reshape(-1,1)
    return x,y,z

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

def OLS_optimal_model(x, y, z, metrics_test, metrics_train, quiet = True, info=""):
    mse_min = metrics_test[1].min()     # The lowest MSE
    at_ = metrics_test[1].argmin()      # The index of mse_min
    best_degree = at_+1                 # The coresponding polynomial degree

    # Find the regression parameters for best_degree
    beta, conf_beta = get_beta(x, y, z, best_degree, "OLS")

    # The corresponding statistics:
    m_test_best = metrics_test[:,at_]
    m_train_best = metrics_train[:,at_]

    if quiet==False:
        print("Optimal model is")
        print ("    Deg  : {}".format(best_degree))
        print ("    RS2  : {:.3f} (train: {:.3f})".format(m_test_best[0], m_train_best[0]))
        print ("    MSE  : {:.3f} (train: {:.3f})".format(m_test_best[1], m_train_best[1]))
        print ("    Var  : {:.3f} (train: {:.3f})".format(m_test_best[2], m_train_best[2]))
        print ("    Bias : {:.3f} (train: {:.3f})".format(m_test_best[3], m_train_best[3]))
        print ("    Beta :", np.array_str(beta.ravel(), precision=2, suppress_small=True))
        print ("    Conf :", np.array_str(conf_beta.ravel(), precision=2, suppress_small=True))
        print("")
        plot.OLS_beta_conf(beta, conf_beta, best_degree, info)

    return beta, best_degree, m_test_best
###############################################################################
#np.random.seed(42)
x, y, z = func.GenerateData(100, 0.01)
#x,y,z = GenDat2()
#X = func.PolyDesignMatrix(x,y,2)

#print(X)

#z_train, z_test, z_fit, z_pred = regression(z, X, "OLS")

d_max = 10
n_bootstraps = 100
k = 5
degrees = np.arange(1, d_max+1)

# Initialise arrays
m_test = np.zeros((4, d_max))       # array to store [R2, mse, var, bias]
m_train = np.zeros((4, d_max))      # array to store [R2, mse, var, bias]

m_test_bs  = np.zeros((4, d_max))       # array to store [R2, mse, var, bias]
m_train_bs = np.zeros((4, d_max))       # array to store [R2, mse, var, bias]

m_test_bs1  = np.zeros((4, d_max))       # array to store [R2, mse, var, bias]
m_train_bs1 = np.zeros((4, d_max))       # array to store [R2, mse, var, bias]

m_test_k  = np.zeros((4, d_max))       # array to store [R2, mse, var, bias]
m_train_k = np.zeros((4, d_max))       # array to store [R2, mse, var, bias]

# Loop over degrees
for i in range(d_max):
    X = func.PolyDesignMatrix(x, y, degrees[i])
    z_train_1, z_test_1, z_fit_1, z_pred_1 = regression(z, X, "OLS", lamb=0)
    m_test[:,i] = func.metrics(z_test_1, z_pred_1, test=True)
    m_train[:,i] = func.metrics(z_train_1, z_fit_1, test=True)

    # With bootstrapping
    z_train_2, z_test_2, z_fit_2, z_pred_2 = func.Bootstrap(x, y, z, degrees[i], n_bootstraps, RegType="OLS", lamb=0)
    m_test_bs[:,i] = func.metrics(z_test_2, z_pred_2, test=True)
    m_train_bs[:,i] = func.metrics(z_train_2, z_fit_2, test=True)

    # With bootstrapping_v1
    z_train_3, z_test_3, z_fit_3, z_pred_3 = func.Bootstrap_v1(x, y, z, degrees[i], n_bootstraps, RegType="OLS", lamb=0)
    m_test_bs1[:,i] = func.metrics(z_test_3, z_pred_3, test=True)
    m_train_bs1[:,i] = func.metrics(z_train_3, z_fit_3, test=True)

    # With kFold
    z_train_3, z_test_3, z_fit_3, z_pred_3 = func.kFold(x, y, z, degrees[i], k, RegType="OLS", lamb=0)
    print(np.shape(z_train_3))
    m_test_k[:,i] = func.metrics(z_test_3, z_pred_3, test=True)
    m_train_k[:,i] = func.metrics(z_train_3, z_fit_3, test=True)

plt.figure()
plt.title("Test")
plt.plot(degrees, m_test[0], label="no_resample")
plt.plot(degrees, m_test_bs[0], label = "Bootstrap")
plt.plot(degrees, m_test_bs1[0], label = "Bootsrap_v1")
plt.plot(degrees, m_test_k[0], label = "kFold")
plt.legend()
plt.ylim((0, 1.2))
plt.show()

plt.figure()
plt.title("Train")
plt.plot(degrees, m_train[0], label="no_resample")
plt.plot(degrees, m_train_bs[0], label = "Bootstrap")
plt.plot(degrees,m_train_bs1[0], label = "Bootstrap_v1")
plt.plot(degrees, m_train_k[0], label = "kFold")
plt.ylim((0, 1.2))
plt.legend()
plt.show()

quit()


# Plotting without resampling
# Recreate figure 2.11 in Hastie (without resampling)
info1 = "n{:.0f}_d{:.0f}".format(len(z), d_max)
plot.OLS_test_train(degrees, m_test[1], m_train[1], err_type ="MSE", info=info1, log=True)

# Find the model with lowest MSE (without resampling)
beta_1, best_degree_1, m_test_best = OLS_optimal_model(x, y, z, m_test, m_train, quiet = False, info=info1)

## Plotting with Bootstrap resampling
info_bs = info1+"_bs{}".format(n_bootstraps)

plot.bias_variance(degrees, m_test_bs[1], m_test_bs[2], m_test_bs[3], "degrees", "OLS", info_bs, log=True)
plot.OLS_test_train(degrees, m_test_bs[1], m_train_bs[1], "MSE", info_bs, log=True)
plot.all_metrics_test_train(degrees, m_test_bs, m_train_bs, "degrees", "OLS", "Bootstrap", info_bs)
beta_2, best_degree_2, m_test_bs_best = OLS_optimal_model(x, y, z, m_test_bs, m_train_bs, quiet = False, info=info_bs)



# Vary the data size and see how the bias-variance behaves

min = 100               # minimum data
max = 500               # maximum data
steps = 100             # steps between
d_1 = best_degree_2     # degree

ndata = np.arange(min, max, steps)  # array of datapoints

m_test_ndata, m_train_ndata = func.OLS_bootstrap_data(ndata, n_bootstraps, d_1)

info_ndata = "min{:.0f}_max{:.0f}_step{:.0f}_d{:.0f}".format(min, max, steps, d_1)
plot.bias_variance(ndata, m_test_ndata[1], m_test_ndata[2], m_test_ndata[3], "data", "OLS", info_ndata, log=True)

# k-fold cross-validation
k = 5   # number of folds

m_test_k = np.zeros((4, d_max))
m_train_k = np.zeros((4, d_max))

for i in range(d_max):
    z_train, z_test, z_fit, z_pred = kFold(x, y, z, degrees[i], k, "OLS")
    m_test_k[:,i] = metrics(z_test, z_pred, test=True)
    m_train_k[:,i] = metrics(z_train, z_fit, test=True)

plt.figure()
plt.plot(degrees, m_test_k[0], label="test")
plt.plot(degrees, m_train_k[0], label="train")
plt.show()
