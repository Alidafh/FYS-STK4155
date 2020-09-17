#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import PolynomialFeatures
from imageio import imread
import general_functions as func
from sklearn.linear_model import Ridge, LinearRegression, Lasso, Ridge

x, y, z = func.GenerateData(100, 0, 1, 0.01, "debug")

x_array = x.ravel()
y_array = y.ravel()
z_array = z.ravel()

# Set up design matrix
p = 5
data = pd.DataFrame.from_dict({"x": x_array, "y": y_array,})
poly = PolynomialFeatures(degree=p).fit(data)
features = pd.DataFrame(poly.transform(data), columns=poly.get_feature_names(data.columns))
print("Design matrix using polynomial of degree {}:".format(p), list(features.columns))


"""
linreg = LinearRegression()
linreg.fit(features, z_array)

zpred = linreg.predict(features)
zplot = zpred.reshape(len(x), len(y))

# Plot the resulting fit beside the original surface
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Franke')

ax = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(x, y, zplot, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Fitted Franke')

plt.show()
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
