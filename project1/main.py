#!/usr/bin/python
import numpy as np
import pandas as pd
from imageio import imread
import general_functions as func
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

x, y, z = func.GenerateData(100, 0, 1, 0.01, "debug")

features = func.PolyDesignMatrix(x,y,degree=2)

x = x.ravel()
y = y.ravel()
z = z.ravel()

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.33)

print(np.shape(x))
print(np.shape(x_train), np.shape(x_test))




#df = pd.DataFrame(features)


"""
terrain1 = imread("datafiles/SRTM_data_Norway_1.tif")

plt.figure()
plt.title("Terrain over Norway 1")
plt.imshow(terrain1, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
"""
