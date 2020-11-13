#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import regression as reg
import tools as tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import SGDRegressor, LinearRegression
import matplotlib.pyplot as plt


#Generate the data for the Franke function
input, y = tools.GenerateDataFranke(ndata=1000, noise_str=0.1)
#------------------------------------------------------------------------------

d_max = 10
degrees = np.arange(1, d_max+1)

# Arrays for storing results
mse_ols = np.zeros((d_max, 2))
r2_ols = np.zeros((d_max, 2))

mse_sgd = np.zeros((d_max, 2))
r2_sgd = np.zeros((d_max, 2))

mse_sgd1 = np.zeros((d_max, 2))
r2_sgd1 = np.zeros((d_max, 2))

for d in range(d_max):
    print("Degree: ", degrees[d])
    X = tools.PolyDesignMatrix(input.T, d=degrees[d])[0]
    X_train, X_test, y_train, y_test = tools.split_scale(X, y)

    # Basic OLS
    ols = reg.OLS()
    ols.fit(X_train, y_train)
    ols_pred = ols.predict(X_test)
    ols_fit = ols.predict(X_train)

    r2_ols[d, 0] =r2_score(y_test, ols_pred)
    r2_ols[d, 1] =r2_score(y_train, ols_fit)

    # SGD OLS: learn_rate: 0.01
    sgd = reg.OLS()
    loss = sgd.SGD(X_train, y_train, n_epochs=100, batch_size=5, learn_rate=0.1)
    sgd_pred = sgd.predict(X_test)
    sgd_fit = sgd.predict(X_train)

    r2_sgd[d, 0] = r2_score(y_test, sgd_pred)
    r2_sgd[d, 1] = r2_score(y_train, sgd_fit)


plt.figure()
plt.grid()
plt.plot(degrees, r2_ols[:,0], label="OLS regression")
plt.plot(degrees, r2_sgd[:,0], label="SGD regression (lr=0.01, M=5)")
plt.plot(degrees, r2_sgd1[:,0], label="Scikit SGDregresson")
plt.legend()
plt.xlabel("Model complexity")
plt.ylabel("R2-score")
plt.show()
