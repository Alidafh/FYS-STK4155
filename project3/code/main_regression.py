#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
NOT DONE JUST AN EXAMPLE
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import config_regression as conf
import pandas as pd


X_train, X_test = conf.X_train, conf.X_test
y_train, y_test = conf.y_train, conf.y_test

model_name = "reg_test"
#model = tf.keras.models.load_model(conf.model_dir+"{:}".format(model_name), custom_objects="coeff_determination")
model = tf.keras.models.load_model(conf.model_dir+"{:}".format(model_name))

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

plt.plot(y_test, y_pred, "o")
plt.xlabel("true")
plt.ylabel("predicted")
plt.show()


log = conf.model_dir+"{:}".format(model_name)
log_data = pd.read_csv(log+'_training.log', sep=',', engine='python')

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(log_data['mean_squared_error'], label='mse')
plt.plot(log_data['val_mean_squared_error'], label = 'val_mse')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend(loc='lower right')

plt.subplot(1,2,2)
plt.plot(log_data['loss'], label='loss')
plt.plot(log_data['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.show()
plt.show()




"""
nlayers = len(model.layers)
layer_outputs = []
layer_names = []
for i in range(nlayers):
    layer = model.layers[i]
    print(i, layer.name, layer.output.shape)
    layer_outputs.append(layer.output)
    layer_names.append(layer.name)
"""
