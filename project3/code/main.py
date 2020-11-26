#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Ignore warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt

from generate import load_data
from CNN import CNN


maps, labels, stats = load_data(PATH="../data/", file="data_(100, 50, 100, 10)_1_0_True_1.csv", slice = None)

print("maps: ", maps.shape)
print("labels", labels.shape)

label_names = ["Clean", "DM"]
"""
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    img = maps[i,:,:,0]
    plt.imshow(img, cmap="inferno")
    plt.xlabel(label_names[int(labels[i])])
plt.show()
"""
X_train, X_test, y_train, y_test = train_test_split(maps, labels, test_size=0.2, random_state=42)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#############################################################################

inputs = {"input_shape": X_train.shape[1:],
          "n_categories": y_train.shape[1],
          "kernel_size": 3,
          "optimizer": "adam",
          "loss": "categorical_crossentropy",
          "metrics": "accuracy",
          "input_activation": "relu",
          "layer_activation": "relu",
          "output_activation": "softmax",
          "n_filters": 64,
          "n_nodes": 32,
          "n_layers": 5,
          "batch_size": 5}

model = CNN(inputs=inputs)


model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20)

from pathlib import Path
Path("tmp").mkdir(parents=True, exist_ok=True)
model.save('tmp/saved_model')

print()
loss, acc = model.evaluate(X_test, y_test, verbose=2)
print()
print("accuracy: {:5.2f}%".format(100 * acc))

print()

y_pred = model.predict(X_test)

#show actual results for the first 4 images in the test set
print(y_test[:4])

#show predictions for the first 4 images in the test set
print(y_pred[:4])

print("First image in test is:    ", y_test[0], label_names[int(y_test[0,1])])
print("First image in predict is: ", y_pred[0], label_names[int(round(y_pred[0,1]))])



"""
plt.figure(figsize=(10,10))
for i in range(20):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    mappp = X_test[i]
    plt.imshow(mappp[:,:,0], cmap="inferno")
    labelx = "{:} (true={:})".format(label_names[int(round(y_pred[i,1]))], label_names[int(y_test[i,1])])
    plt.xlabel(labelx)
plt.show()
"""
