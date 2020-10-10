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

##############################################################################
FOLDER = sys.argv[1]
###############################################################################
if FOLDER == "franke": x, y, z = func.GenerateData(100, 0.01)
if FOLDER == "data": x, y, z = func.map_to_data("datafiles/SRTM_data_Norway_1.tif")
################################################################################

print("#######################################################################")
print("                                Ridge                                  ")
print("#######################################################################")

print("\nFinding the optimal combination of lambda and degree: \n")

d_max = 10
n_bootstraps = 100
k = 5
nlamb = 10

degrees = np.arange(1, d_max+1)
lambdas = np.logspace(-3, 0, nlamb)

# arrays to store things in
lamb_1 = np.zeros(d_max)    # optimal lambdas as a function of degrees
lamb_bs = np.zeros(d_max)
lamb_k = np.zeros(d_max)

m_1 = np.zeros((4, d_max))
m_bs = np.zeros((4, d_max))
m_k = np.zeros((4, d_max))

# Loop over degrees
for j in range(d_max):
    d_ridge = degrees[j]

    m_test = np.zeros((4, nlamb))         # array to store [R2, mse, var, bias]
    m_train = np.zeros((4, nlamb))        # array to store [R2, mse, var, bias]

    m_test_bs  = np.zeros((4, nlamb))     # array to store [R2, mse, var, bias]
    m_train_bs = np.zeros((4, nlamb))     # array to store [R2, mse, var, bias]

    m_test_k  = np.zeros((4, nlamb))      # array to store [R2, mse, var, bias]
    m_train_k = np.zeros((4, nlamb))      # array to store [R2, mse, var, bias]

    # Loop over lambdas
    for i in range(nlamb):

        # Without resampling
        X = func.PolyDesignMatrix(x, y, d_ridge)
        z_train_1, z_test_1, z_fit_1, z_pred_1 = func.regression(z, X, "RIDGE", lamb=lambdas[i])
        m_test[:,i] = func.metrics(z_test_1, z_pred_1, test=True)
        m_train[:,i] = func.metrics(z_train_1, z_fit_1, test=True)

        # With bootstrapping
        z_train_2, z_test_2, z_fit_2, z_pred_2 = func.Bootstrap(x, y, z, d_ridge, n_bootstraps, RegType="RIDGE", lamb=lambdas[i])
        m_test_bs[:,i] = func.metrics(z_test_2, z_pred_2, test=True)
        m_train_bs[:,i] = func.metrics(z_train_2, z_fit_2, test=True)

        # With kFold
        z_train_4, z_test_4, z_fit_4, z_pred_4 = func.kFold(x, y, z, d_ridge, k, shuffle=True, RegType="RIDGE", lamb=lambdas[i])
        m_test_k[:,i] = func.metrics(z_test_4, z_pred_4, test=True)
        m_train_k[:,i] = func.metrics(z_train_4, z_fit_4, test=True)

    # Find the lambda value that has the lowest Mean Squared Error,
    # and store both the lambda and the corresponding metrics
    lamb_1[j],  m_1[:,j]  = func.optimal(lambdas, m_test)
    lamb_bs[j], m_bs[:,j] = func.optimal(lambdas, m_test_bs)
    lamb_k[j],  m_k[:,j]  = func.optimal(lambdas, m_test_k)


print("###############################################")
print("         Without Resampling                    ")
print("###############################################")
# Find the degree with the lowest mean squared error
best_degree_ridge, best_m_ridge = func.optimal(degrees, m_1)
best_lamb_ridge = lamb_1[best_degree_ridge-1]

info_ridge1 = "n{:.0f}_deg{:.0f}_lamb{:.4f}".format(len(z), best_degree_ridge, best_lamb_ridge)
func.print_plot_modelparams(x, y, z, m_test=best_m_ridge, d =best_degree_ridge, lamb=best_lamb_ridge, rType = "RIDGE", quiet = False, info=info_ridge1, f=FOLDER)

print("###############################################")
print("             With Bootstrap                    ")
print("###############################################")

best_degree_ridge_bs, best_m_ridge_bs = func.optimal(degrees, m_bs)
best_lamb_ridge_bs = lamb_bs[best_degree_ridge_bs-1]

info_ridge_bs = "n{:.0f}_deg{:.0f}_lamb{:.4f}_bs{:}".format(len(z), best_degree_ridge_bs, best_lamb_ridge_bs, n_bootstraps)
func.print_plot_modelparams(x, y, z, best_m_ridge_bs, best_degree_ridge_bs, best_lamb_ridge_bs, rType = "RIDGE", quiet = False, info=info_ridge_bs, f=FOLDER)

print("###############################################")
print("              With kFold                       ")
print("###############################################")

best_degree_ridge_k, best_m_ridge_k = func.optimal(degrees, m_k)
best_lamb_ridge_k = lamb_k[best_degree_ridge_k-1]

info_ridge_k = "n{:.0f}_deg{:.0f}_lamb{:.4f}_kFold{:}".format(len(z), best_degree_ridge_k, best_lamb_ridge_k, k)
func.print_plot_modelparams(x, y, z, best_m_ridge_k, best_degree_ridge_k, best_lamb_ridge_k, rType = "RIDGE", quiet = False, info=info_ridge_k, f=FOLDER)


print("Bias variance relations\n")


m_test = np.zeros((4, d_max))           # array to store [R2, mse, var, bias]
m_train = np.zeros((4, d_max))           # array to store [R2, mse, var, bias]

m_test_k = np.zeros((4, d_max))           # array to store [R2, mse, var, bias]
m_train_k = np.zeros((4, d_max))           # array to store [R2, mse, var, bias]

m_test_bs1 = np.zeros((4, d_max))           # array to store [R2, mse, var, bias]
m_train_bs1 = np.zeros((4, d_max))          # array to store [R2, mse, var, bias]

m_test_bs2  = np.zeros((4, d_max))       # array to store [R2, mse, var, bias]
m_train_bs2 = np.zeros((4, d_max))       # array to store [R2, mse, var, bias]

m_test_bs3  = np.zeros((4, d_max))       # array to store [R2, mse, var, bias]
m_train_bs3 = np.zeros((4, d_max))       # array to store [R2, mse, var, bias]

# Loop over degrees
for i in range(d_max):
    # Without resampling for lambda = 0.01
    X_ = func.PolyDesignMatrix(x, y, degrees[i])
    z_train, z_test, z_fit, z_pred = func.regression(z, X_, "RIDGE", lamb=0.01)
    m_test[:,i] = func.metrics(z_test, z_pred, test=True)
    m_train[:,i] = func.metrics(z_train, z_fit, test=True)

    # With kFold for lambda 0.01
    z_train_k, z_test_k, z_fit_k, z_pred_k = func.kFold(x, y, z, degrees[i], k, shuffle=True, RegType="RIDGE", lamb=0.01)
    m_test_k[:,i] = func.metrics(z_test_k, z_pred_k, test=True)
    m_train_k[:,i] = func.metrics(z_train_k, z_fit_k, test=True)

    # With bootstrapping for lambda = 1
    z_train_1, z_test_1, z_fit_1, z_pred_1 = func.Bootstrap(x, y, z, degrees[i], n_bootstraps, RegType="RIDGE", lamb=1)
    m_test_bs1[:,i] = func.metrics(z_test_1, z_pred_1, test=True)
    m_train_bs1[:,i] = func.metrics(z_train_1, z_fit_1, test=True)

    # With bootstrapping for lambda = 0.01
    z_train_2, z_test_2, z_fit_2, z_pred_2 = func.Bootstrap(x, y, z, degrees[i] , n_bootstraps, RegType="RIDGE", lamb=0.01)
    m_test_bs2[:,i] = func.metrics(z_test_2, z_pred_2, test=True)
    m_train_bs2[:,i] = func.metrics(z_train_2, z_fit_2, test=True)

    # With bootstrapping for lambda = 0.0
    z_train_3, z_test_3, z_fit_3, z_pred_3 = func.Bootstrap(x, y, z, degrees[i], n_bootstraps, RegType="RIDGE", lamb=0.0)
    m_test_bs3[:,i] = func.metrics(z_test_3, z_pred_3, test=True)
    m_train_bs3[:,i] = func.metrics(z_train_3, z_fit_3, test=True)

print("###############################################")
print("Bias-variance vs degrees for different lambdas ")
print("###############################################")

info_r_bs1 = "n{:.0f}_bs{:}_lamb{:.4f}".format(len(z), n_bootstraps, 1)
info_r_bs2 = "n{:.0f}_bs{:}_lamb{:.4f}".format(len(z), n_bootstraps, 0.01)
info_r_bs3 = "n{:.0f}_bs{:}_lamb{:.4f}".format(len(z), n_bootstraps, 0.0)

plot.bias_variance(degrees, m_test_bs1[1], m_test_bs1[2], m_test_bs1[3], x_type="degrees", RegType ="RIDGE", info=info_r_bs1, log=True, FOLDER=FOLDER)
plot.bias_variance(degrees, m_test_bs2[1], m_test_bs2[2], m_test_bs2[3], x_type="degrees", RegType ="RIDGE", info=info_r_bs2, log=True, FOLDER=FOLDER)
plot.bias_variance(degrees, m_test_bs3[1], m_test_bs3[2], m_test_bs3[3], x_type="degrees", RegType ="RIDGE", info=info_r_bs3, log=True, FOLDER=FOLDER)

print("###############################################")
print("  Comparisons between k-Fold and Bootstrap     ")
print("###############################################")

info_comp = "n{:}_lambda{:}_bs{:}_k{:}".format(len(z), 0.01, d_max, n_bootstraps, k)
plot.compare_MSE(degrees, m_test[1], m_test_bs2[1], m_test_k[1], rType = "RIDGE", lamb=0.01, info=info_comp, log=True, FOLDER=FOLDER)
plot.compare_R2(degrees, [m_test[0], m_train[0]], [m_test_bs2[0], m_train_bs2[0]], [m_test_k[0],m_train_k[0]] , rType = "RIDGE", info=info_comp, FOLDER=FOLDER)

print("###############################################")
print(" Bias-variance vs lambdas for different degrees")
print("###############################################")

# Initialize ararys
m_test = np.zeros((4, nlamb))           # array to store [R2, mse, var, bias]
m_train = np.zeros((4, nlamb))           # array to store [R2, mse, var, bias]

m_test_bs1 = np.zeros((4, nlamb))           # array to store [R2, mse, var, bias]
m_train_bs1 = np.zeros((4, nlamb))          # array to store [R2, mse, var, bias]

m_test_bs2  = np.zeros((4, nlamb))       # array to store [R2, mse, var, bias]
m_train_bs2 = np.zeros((4, nlamb))       # array to store [R2, mse, var, bias]

m_test_bs3  = np.zeros((4, nlamb))       # array to store [R2, mse, var, bias]
m_train_bs3 = np.zeros((4, nlamb))       # array to store [R2, mse, var, bias]

# Loop over lambdas
for i in range(nlamb):
    # Without resampling for d = 6
    X_ = func.PolyDesignMatrix(x, y, 6)
    z_train, z_test, z_fit, z_pred = func.regression(z, X_, "RIDGE", lamb=lambdas[i])
    m_test[:,i] = func.metrics(z_test, z_pred, test=True)
    m_train[:,i] = func.metrics(z_train, z_fit, test=True)

    # With kFold for d = 6
    z_train_k, z_test_k, z_fit_k, z_pred_k = func.kFold(x, y, z, 6, k, shuffle=True, RegType="RIDGE", lamb=lambdas[i])
    m_test_k[:,i] = func.metrics(z_test_k, z_pred_k, test=True)
    m_train_k[:,i] = func.metrics(z_train_k, z_fit_k, test=True)

    # With bootstrapping for d = 2
    z_train_1, z_test_1, z_fit_1, z_pred_1 = func.Bootstrap(x, y, z, 2, n_bootstraps, RegType="RIDGE", lamb=lambdas[i])
    m_test_bs1[:,i] = func.metrics(z_test_1, z_pred_1, test=True)
    m_train_bs1[:,i] = func.metrics(z_train_1, z_fit_1, test=True)

    # With bootstrapping for d = 6
    z_train_2, z_test_2, z_fit_2, z_pred_2 = func.Bootstrap(x, y, z, 6 , n_bootstraps, RegType="RIDGE", lamb=lambdas[i])
    m_test_bs2[:,i] = func.metrics(z_test_2, z_pred_2, test=True)
    m_train_bs2[:,i] = func.metrics(z_train_2, z_fit_2, test=True)

    # With bootstrapping for d = 9
    z_train_3, z_test_3, z_fit_3, z_pred_3 = func.Bootstrap(x, y, z, 9, n_bootstraps, RegType="RIDGE", lamb=lambdas[i])
    m_test_bs3[:,i] = func.metrics(z_test_3, z_pred_3, test=True)
    m_train_bs3[:,i] = func.metrics(z_train_3, z_fit_3, test=True)


print("###############################################")
print("             With Bootstrap                    ")
print("###############################################")

info_ridge_d2 = "n{:.0f}_d{:.0f}_bs{:.0f}".format(len(z), 2, n_bootstraps)
info_ridge_d6 = "n{:.0f}_d{:.0f}_bs{:.0f}".format(len(z), 6, n_bootstraps)
info_ridge_d9 = "n{:.0f}_d{:.0f}_bs{:.0f}".format(len(z), 9, n_bootstraps)

plot.bias_variance(lambdas, m_test_bs1[1], m_test_bs1[2], m_test_bs1[3], x_type="lambda", RegType ="RIDGE", info=info_ridge_d2, log=True, FOLDER=FOLDER)
plot.bias_variance(lambdas, m_test_bs2[1], m_test_bs2[2], m_test_bs2[3], x_type="lambda", RegType ="RIDGE", info=info_ridge_d6, log=True, FOLDER=FOLDER)
plot.bias_variance(lambdas, m_test_bs3[1], m_test_bs3[2], m_test_bs3[3], x_type="lambda", RegType ="RIDGE", info=info_ridge_d9, log=True, FOLDER=FOLDER)
