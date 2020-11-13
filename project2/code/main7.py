#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import regression as reg
import numpy as np
import mnist_loader
import tools
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
"""
X, y = tools.GenerateDataLine(100)
X_train, X_test, y_train, y_test = tools.split_scale(X,y)

ols = reg.OLS()
ols.fit(X_train, y_train)
y_pred = ols.predict(X_test)

print("Ordinary Least Squares:")
print("-----------------------------------")
print("r2:", ols.r2score(X_test,y_test))
print("mse:", ols.mse(X_test, y_test))
print()

sgd = reg.OLS()
sgd.SGD(X_train, y_train, n_epochs=100, batch_size=5, learn_rate=0.01)
y_pred2 = sgd.predict(X_test)

print("Stochastic Gradient Descent")
print("-----------------------------------")
print("r2:", sgd.r2score(X_test, y_test))
print("mse:", sgd.mse(X_test, y_test))
print()

plt.figure()
plt.grid()
plt.plot(X_train[:,1], y_train, color="tab:gray", marker=".", linewidth=0, markersize=12, label="Train data")
plt.plot(X_test[:,1], y_test, "ko", label="Test data")
plt.plot(X_test[:,1], y_pred, "tab:red", label="OLS fit")
plt.plot(X_test[:,1], y_pred2, "tab:blue", label="SGD fit")
plt.legend()
plt.show()


ols, sgd = None, None
"""
##############################################################################
print("############################################################\n")
input, y = tools.GenerateDataFranke(ndata=100, noise_str=0.1)
X,p = tools.PolyDesignMatrix(input.T, d=6)

ols = reg.OLS()
ols.fit(X, y)
y_pred = ols.predict(X)

print("Ordinary Least Squares:")
print("-----------------------------------")
print("r2:", ols.r2score(X,y))
print("mse:", ols.mse(X, y))
print()

sgd = reg.OLS()
sgd.SGD(X, y, n_epochs=100, batch_size=5, learn_rate = 0.1)
y_pred2 = sgd.predict(X)

print("Stochastic Gradient Descent")
print("-----------------------------------")
print("r2:", sgd.r2score(X,y))
print("mse:", sgd.mse(X, y))
print()

x1, x2, y1 = tools.reshape_franke(input, y)
x1, x2, y_pred = tools.reshape_franke(input, y_pred)
x1, x2, y_pred2 = tools.reshape_franke(input, y_pred2)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x1, x2, y1, s=0.5, c="k")
ax.plot_surface(x1, x2, y_pred, color="b")
ax.plot_surface(x1, x2, y_pred2, color="r")
plt.show()
"""
ols, sgd = None, None
###############################################################################
print("############################################################\n")
input, y = tools.GenerateDataFranke(ndata=100, noise_str=0.1)
X, p = tools.PolyDesignMatrix(input.T, d=6)
X_train, X_test, y_train, y_test = tools.split_scale(X, y)

ols = reg.OLS()
ols.fit(X_train, y_train)

print("Ordinary Least Squares:")
print("-----------------------------------")
print("r2:", ols.r2score(X_test,y_test))
print("mse:", ols.mse(X_test, y_test))
print()

sgd = reg.OLS()
loss = sgd.SGD(X_train, y_train, n_epochs=100, batch_size=5)

print("Stochastic Gradient Descent")
print("-----------------------------------")
print("r2:", sgd.r2score(X_test, y_test))
print("mse:", sgd.mse(X_test, y_test))
print()
"""
