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


maps, labels, stats = load_data(PATH="../data/", file="data_(2000, 50, 50, 3)_1_0_True_.csv", slice = None)

X_train, X_test, y_train, y_test = train_test_split(maps, labels, test_size=0.2, random_state=42)
X_train = (X_train - np.mean(X_train))/np.std(X_train)
X_test = (X_test-np.mean(X_train))/np.std(X_train)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print()

label_names = ["Clean", "DM"]
"""
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    img = maps[i,:,:,5]
    plt.imshow(img, cmap="inferno")
    plt.xlabel(label_names[int(labels[i])])
plt.show()
"""
print()
print(X_train.shape[1:])
print(y_train.shape[1])

#############################################################################

inputs = {"input_shape": X_train.shape[1:],
          "n_categories": len(label_names), #y_train.shape[1]
          "kernel_size": 3,
          "n_filters": X_train.shape[1],
          "n_nodes": 2*X_train.shape[1],
          "n_layers": 3,
          "optimizer": "adam",
          "learn_rate": 1e-4,
          "loss": "mean_squared_error",
          "metrics": "accuracy",
          "input_activation": "relu",
          "layer_activation": "relu",
          "output_activation": "sigmoid"}


model = CNN(inputs=inputs)
print(model.summary())

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20)

from pathlib import Path
Path("tmp").mkdir(parents=True, exist_ok=True)
model.save('tmp/saved_model')

plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()


test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print("accuracy: {:5.2f}%".format(100 * test_acc))

y_pred = model.predict(X_test)

plt.figure(figsize=(10,10))
for i in range(y_test.shape[0]):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_test[i,:,:,5], cmap="inferno")
    labelx = "{:} (true={:})".format(label_names[int(round(y_pred[i,1]))], label_names[int(y_test[i,1])])
    plt.xlabel(labelx)
plt.show()
