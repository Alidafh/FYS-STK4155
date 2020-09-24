#!/usr/bin/python
import numpy as np
import pandas as pd
from imageio import imread
import general_functions as func
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

x, y, z = func.GenerateData(100, 0, 1, 0.01, "debug")
X_features = func.PolyDesignMatrix(x, y, degree=2)
X_train, X_test, z_train, z_test = train_test_split(X_features, z.ravel(), test_size=0.33)
beta, z_train_predict = func.OLS(z_train, X_train)
R2, MSE, var = func.metrics(z_train, z_train_predict)




"""
terrain1 = imread("datafiles/SRTM_data_Norway_1.tif")

plt.figure()
plt.title("Terrain over Norway 1")
plt.imshow(terrain1, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
"""
