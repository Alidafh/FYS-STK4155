#!/usr/bin/python
import numpy as np
import pandas as pd
from imageio import imread
import general_functions as func

x, y, z = func.GenerateData(4, 0, 1, 0.01, "debug")

features = func.PolyDesignMatrix(x,y,degree=2)
df = pd.DataFrame(features)
print(df)
