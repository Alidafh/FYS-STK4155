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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tools import r2_score
import config_regression as conf
import pandas as pd
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["tab:blue", "tab:green", "tab:red", "tab:purple", "tab:orange", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"])


# get the saved model
name = "reg10true3"
model_name = conf.model_dir+name
model = tf.keras.models.load_model(model_name, custom_objects={"r2_score": r2_score})
model.summary()


# easier to use without writing conf all the time
X_train, X_test = conf.X_train, conf.X_test
y_train, y_test = conf.y_train, conf.y_test

# print the loss and r2 score of the model
loss, metric = model.evaluate(X_test, y_test, verbose=0)
print(f"Loss (MSE): {loss:.4f} - r2_score: {metric:.4f}")
print()

# predict using the test data and normalize like they do in the paper
y_pred = model.predict(X_test)
y_pred = y_pred/y_pred.max()

# Plot predicted vs true
fig = plt.figure()
plt.plot(y_test, y_pred, marker="o", linestyle="None", color="tab:blue", label="data")
plt.plot(y_test, y_test, linestyle="dashed", color="k", label="Perfect prediction")
plt.xlabel("$f_{dms}$ true")
plt.ylabel("$f_{dms}$ predicted")
plt.legend()
fig.savefig(f"../figures/{name}_true_vs_predict.png")
#plt.show()


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
fig.savefig(f"../figures/{name}_history.png")
#plt.show()


# Choose an image from the test set and display it
img = np.expand_dims(X_test[0], axis=0)
f_dms = y_test[0]

fig = plt.figure()
plt.imshow(img[0,:,:,0], cmap="inferno")
plt.xticks([])
plt.yticks([])
plt.xlabel("$f_{dms}:$"+f"{f_dms[0]:.4f}")
fig.savefig(f"../figures/{name}_test_img.png")


# get the layers
layer_outputs = []
layer_names = []

for i in range(len(model.layers)):
    layer = model.layers[i]
    layer_outputs.append(layer.output)
    layer_names.append(layer.name)

# Set up a model
activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)

# make a prediction using the model and the image as input
pred_img = activation_model.predict(img)

# the first 6 layers are convolutional and max pooling layers
conv_pool = pred_img[:6]
n_layers = len(conv_pool)
n_filters = 5

# display the first 5 filters of each layer
fig, ax = plt.subplots(nrows=n_filters, ncols=n_layers, figsize=(20,20))
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

for i in range(n_layers):
    map = conv_pool[i]
    for j in range(n_filters):
        im = ax[j,i].imshow(map[0,:,:,j], cmap="inferno")
        ax[0,i].set_title(layer_names[i])
        ax[j,0].set_ylabel("filter {:}".format(j), size='large')

fig.colorbar(im, ax=ax.flat)
fig.savefig(f"../figures/{name}_filters_test_img.png")
plt.show()

"""
# Plot the cross validation
fig, ax = plt.subplots(nrows=2, ncols=1, sharex="col", sharey=False, constrained_layout=True)
ax[0].set_ylabel("Loss MSE")
ax[1].set_ylabel("R2-score")

kfold_path = conf.model_dir+"/kFold_"+conf.type+"/"+name
num_folds = 5

c = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",]

for i in range(1,num_folds+1):
    mn = kfold_path+f"_{i}"
    log = pd.read_csv(mn+"_training.log", sep=",", engine="python")

    ax[0].plot(log["loss"],         color=c[i-1], label=f"Fold # {i}")
    ax[0].plot(log["val_loss"],     color=c[i-1], linestyle="dashed")
    ax[1].plot(log["r2_score"],     color=c[i-1], label=f"Fold # {i}")
    ax[1].plot(log["val_r2_score"], color=c[i-1], linestyle="dashed")

plt.xlabel("Epoch")
ax[0].legend(loc = "upper right")

fig.savefig(f"../figures/{name}_kfold.png")
#plt.show()
"""

























"""
# get the convolutional layers
layer_outputs = []
layer_names = []

for i in range(len(model.layers)):
    layer = model.layers[i]
    if "conv" in layer.name or "pool" in layer.name:
        print(layer.name)
        layer_outputs.append(layer.output)
        layer_names.append(layer.name)

activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)

# make a prediction usin the model
pred_img = activation_model.predict(img)

n_layers = len(pred_img)
n_filters = 5

fig, ax = plt.subplots(nrows=n_filters, ncols=n_layers, figsize=(20,20))
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

for i in range(n_layers):
    map = pred_img[i]
    for j in range(n_filters):
        im = ax[j,i].imshow(map[0,:,:,j], cmap="inferno")
        ax[0,i].set_title(layer_names[i])
        ax[j,0].set_ylabel("filter {:}".format(j), size='large')

fig.colorbar(im, ax=ax.flat)
plt.show()


"""


















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


(X_train, y_train), (X_test, y_test) = gce(d3=False, seed=42, scale=True)
label_names = ["Clean", "DM"]

img = X_test[0]
img_type = label_names[int(y_test[0].argmax())]
img = np.expand_dims(img, axis=0)

plt.imshow(img[0,:,:,9], cmap="inferno")
plt.xlabel(img_type)
plt.show()

model = get_model(model_name="GCE")

nlayers = len(model.layers)

layer_outputs = []
layer_names = []
for i in range(nlayers):
    layer = model.layers[i]
    if "conv2d" not in layer.name: continue
    print(i, layer.name, layer.output.shape)
    layer_outputs.append(layer.output)
    layer_names.append(layer.name)


activation_model = Model(inputs=model.input, outputs=layer_outputs)
maps = activation_model.predict(img)

n_layers = len(maps)
n_filters = 5

fig, ax = plt.subplots(nrows=n_filters, ncols=n_layers, figsize=(20,20))
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

for i in range(n_layers):
    map = maps[i]
    for j in range(n_filters):
        im = ax[j,i].imshow(map[0,:,:,j], cmap="inferno")
        ax[0,i].set_title(layer_names[i])
        ax[j,0].set_ylabel("filter {:}".format(j), size='large')


fig.colorbar(im, ax=ax.flat)
plt.show()
"""
