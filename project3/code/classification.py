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
from tools import r2_score, history_classification
import config_classification as conf
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, plot_confusion_matrix


# get the saved model
name = "clas10one"
model_name = conf.model_dir+name
model = tf.keras.models.load_model(model_name)
model.summary()

# easier to use without writing conf all the time
X_train, X_test = conf.X_train, conf.X_test
y_train, y_test = conf.y_train, conf.y_test

# print the loss and accuracy of the model
loss, metric = model.evaluate(X_test, y_test, verbose=0)
print(f"Loss (MSE): {loss:.4f} - Accuracy: {metric*100:.4f}%")
print()

y_pred = model.predict(X_test)

# get roc and auc
false_pr, true_pr, thresholds = roc_curve(y_test, y_pred)
auc = auc(false_pr, true_pr)

fig = plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(false_pr, true_pr, label=f"AUC = {auc:.3f}")
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc='best')
fig.savefig(f"../figures/{name}_roc.png")


# get the log file
log_data = pd.read_csv(model_name+'_training.log', sep=',', engine='python')

# plot the loss and accuracy vs number of epochs
fig, ax = plt.subplots(nrows=2, ncols=1, sharex="col", sharey=False, constrained_layout=True)

c = ["tab:blue", "tab:green"]

ax[0].set_ylabel("Loss (catagorical crossentropy)")
ax[0].plot(log_data['loss'], color=c[0], label='training')
ax[0].plot(log_data['val_loss'], color=c[1], label = 'validation')
ax[0].legend(loc = 'upper right')

ax[1].set_ylabel("Accuracy")
ax[1].plot(log_data["accuracy"], color=c[0], label="training")
ax[1].plot(log_data['val_accuracy'], color=c[1], label = 'validation')
plt.xlabel('Epoch')
fig.savefig(f"../figures/{name}_history.png")


# plot the first 25 images of the test set and the predicted categories
fig = plt.figure(figsize=(10,10))
slice=0
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_test[i][:,:,slice], cmap="inferno")
    n = int(y_test[i][0])
    m = round(y_pred[i][0])
    l = "{:} (true={:})".format(conf.label_names[m], conf.label_names[n])
    plt.xlabel(l)

fig.savefig(f"../figures/{name}_results.png")


c_matrix = confusion_matrix(y_test.ravel(), y_pred.ravel())
print(c_matrix)

quit()
# Plot the cross validation
fig, ax = plt.subplots(nrows=2, ncols=1, sharex="col", sharey=False, constrained_layout=True)
ax[0].set_ylabel("Loss")
ax[1].set_ylabel("Accuracy")

kfold_path = conf.model_dir+"/kFold_"+conf.type+"/"+name
num_folds = 5

c = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",]

for i in range(1,num_folds+1):
    mn = kfold_path+f"_{i}"
    log = pd.read_csv(mn+"_training.log", sep=",", engine="python")

    ax[0].plot(log["loss"],         color=c[i-1], label=f"Fold # {i}")
    ax[0].plot(log["val_loss"],     color=c[i-1], linestyle="dashed")
    ax[1].plot(log["accuracy"],     color=c[i-1], label=f"Fold # {i}")
    ax[1].plot(log["val_accuracy"], color=c[i-1], linestyle="dashed")

plt.xlabel("Epoch")
ax[0].legend(loc = "upper right")

fig.savefig(f"../figures/{name}_kfold.png")

plt.show()
