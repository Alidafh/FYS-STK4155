#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import regression as reg
import numpy as np
import mnist_loader
import tools
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

input, y = tools.GenerateDataFranke(ndata=1000, noise_str=0.1)
X, p = tools.PolyDesignMatrix(input.T, d=7)
#X, y = tools.GenerateDataLine(100)

X_train, X_test, y_train, y_test = tools.split_scale(X, y)

model = reg.OLS()
loss = model.SGD(X_train, y_train, n_epochs=100, batch_size=2, tol=1e-4, prin=True)

print("Beta : ", model.beta, sep='\n')
print("r2 : ", model.r2score(X_test, y_test))
print("mse: ", model.mse(X_test, y_test), '\n')
