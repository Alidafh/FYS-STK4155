#!/usr/bin/python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import PolynomialFeatures
from imageio import imread
import general_functions as func
from sklearn.linear_model import Ridge, LinearRegression, Lasso, Ridge

x, y, z = func.GenerateData(4, 0, 1, 0.01, "debug")

features = func.PolyDesignMatrix(x,y,degree=2)
df = pd.DataFrame(features)
print(df)
