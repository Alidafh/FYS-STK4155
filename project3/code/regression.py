#!/usr/bin/env python3
# -*- coding: utf-8 -*

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tools import r2_score, history_regression, test_predict, cross_validation_regression
import config_regression as conf
import pandas as pd

# easier to use without writing conf all the time
X_train, X_test = conf.X_train, conf.X_test
y_train, y_test = conf.y_train, conf.y_test

# get the saved model
model_name = conf.model_dir+"reg2"
model = tf.keras.models.load_model(model_name, custom_objects={"r2_score": r2_score})

# print summary
model.summary()

# print the loss and r2 score
loss, r2 = model.evaluate(X_test, y_test, verbose=0)
print("loss(MSE): %.4f" %loss, "r2-score:  %.4f" %r2, sep='\n')
print()

# get the log file
log_data = pd.read_csv(model_name+'_training.log', sep=',', engine='python')

# plot the loss and r2 vs number of epochs
history_regression(log_data, title="reg1_history")

# predict using the test data
y_pred = model.predict(X_test)

y_pred = y_pred/y_pred.max()

# plot the true value vs the predicted
test_predict(y_test, y_pred, title="reg1_test_predict")

cross_validation_regression(name="reg2", conf=conf, title="kFold_reg2")

plt.show()


"""
residuals = y_test - y_pred
RSS = residuals.T @ residuals
ndf = len(y_pred)-1

nlayers = len(model.layers)
layer_outputs = []
layer_names = []
for i in range(nlayers):
    layer = model.layers[i]
    print(i, layer.name, layer.output.shape)
    layer_outputs.append(layer.output)
    layer_names.append(layer.name)



y_pred = model.predict(X_test)
print(y_pred.shape, y_test.shape)
y_pred = y_pred.ravel()
y_test = y_test.ravel()
print(y_pred.shape, y_test.shape)

residual = y_pred - y_test

Pdiff = (residual/y_test)*100
absPdiff = np.abs(Pdiff)
mean = np.mean(absPdiff)
std = np.std(absPdiff)
print()
print("mean absolute percent difference: {:5.2f}%".format(mean))
print("std absolute percent difference: {:5.2f}%".format(std))

"""
