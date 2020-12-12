#!/usr/bin/env python3
# -*- coding: utf-8 -*

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tools import r2_score, history_regression, test_predict, cross_validation_regression
import config_regression as conf
import pandas as pd
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["tab:blue", "tab:green", "tab:red", "tab:purple", "tab:orange", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"])



# get the saved model
model_name = conf.model_dir+"reg2"
model_reg2 = tf.keras.models.load_model(model_name, custom_objects={"r2_score": r2_score})
model_reg2.summary()


# easier to use without writing conf all the time
X_train, X_test = conf.X_train, conf.X_test
y_train, y_test = conf.y_train, conf.y_test

# print the loss and r2 score of the model
loss, r2 = model_reg2.evaluate(X_test, y_test, verbose=0)
print("loss(MSE): %.4f" %loss, "r2-score:  %.4f" %r2, sep='\n')
print()

# predict using the test data and normalize like they do in the paper
y_pred = model_reg2.predict(X_test)
y_pred = y_pred/y_pred.max()

fig = plt.figure()
plt.plot(y_test, y_pred, marker="o", linestyle="None", color="tab:blue", label="data")
plt.plot(y_test, y_test, linestyle="dashed", color="k", label="Perfect prediction")
plt.xlabel("$f_{dms}$ true")
plt.ylabel("$f_{dm}$ predicted")
plt.legend()
fig.savefig("../figures/reg2_true_vs_predict.png")
plt.show()


# get the log file
log_data = pd.read_csv(model_name+'_training.log', sep=',', engine='python')

# plot the loss and r2 vs number of epochs
fig, ax = plt.subplots(nrows=2, ncols=1, sharex="col", sharey=False, constrained_layout=True)
c = ["tab:blue", "tab:green"]
ax[0].set_ylabel("Loss MSE")
ax[0].plot(log_data['loss'], color=c[0], label='training')
ax[0].plot(log_data['val_loss'], color=c[1], label = 'validation')
ax[0].legend(loc = 'upper right')
ax[1].set_ylabel("R2-score")
ax[1].plot(log_data["r2_score"], color=c[0], label="training")
ax[1].plot(log_data['val_r2_score'], color=c[1], label = 'validation')
plt.xlabel('Epoch')
fig.savefig("../figures/reg2_history.png")
plt.show()


# Plot the cross validation
fig, ax = plt.subplots(nrows=2, ncols=1, sharex="col", sharey=False, constrained_layout=True)
ax[0].set_ylabel("Loss MSE")
ax[1].set_ylabel("R2-score")

kfold_path="/kFold_"+conf.type+"/"
num_folds = 5
c = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",]

for i in range(1,num_folds+1):
    mn = conf.model_dir+kfold_path+"reg2"+str(i)
    log = pd.read_csv(mn+"_training.log", sep=",", engine="python")

    ax[0].plot(log["loss"],         color=c[i-1], label=f"Fold # {i}")
    ax[0].plot(log["val_loss"],     color=c[i-1], linestyle="dashed")
    ax[1].plot(log["r2_score"],     color=c[i-1], label=f"Fold # {i}")
    ax[1].plot(log["val_r2_score"], color=c[i-1], linestyle="dashed")

plt.xlabel("Epoch")
ax[0].legend(loc = "upper right")

fig.savefig("../figures/reg2_kfold.png")

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
