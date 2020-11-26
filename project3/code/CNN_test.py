#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ignore warnings:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

  Level | Level for Humans | Level Description
 -------|------------------|------------------------------------
  0     | DEBUG            | [Default] Print all messages
  1     | INFO             | Filter out INFO messages
  2     | WARNING          | Filter out INFO & WARNING messages
  3     | ERROR            | Filter out all messages
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from generate import read_data

from sklearn.model_selection import train_test_split

PATH = "../data/"
dim = (50,100,10)
n = 40

data = read_data(PATH, dim, n_maps_in_file = n, combine=True, ddf=False, shuf=True)

labels = data[:,0]
maps = data[:,1:]

print(labels.shape)
print(maps.shape)

maps = maps.reshape(40, 50, 100, 10)

print(labels.shape)
print(maps.shape)

maps = maps[:,:,:,5]

print(labels.shape)
print(maps.shape)
print()

label_names = ["Clean", "DM"]

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(maps[i], cmap="inferno")
    plt.xlabel(label_names[int(labels[i])])
plt.show()

maps = maps.reshape(40,50,100,1)
print(labels.shape)
print(maps.shape)
print()

train_size = 0.8
test_size = 1 - train_size
X_train, X_test, y_train, y_test = train_test_split(maps, labels, train_size=train_size, test_size=test_size)

print()
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

print()
print(y_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_test)

print()
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print()

print(y_train)
print()


def create_model():
    model = Sequential()

    model.add(layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(50, 100, 1)))
    model.add(layers.Conv2D(32, kernel_size=3, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation='softmax'))

    #compile model using accuracy as a measure of model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model()
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


print()
print("First image in test is:    ", y_test[0], label_names[int(y_test[0,1])])
print("First image in predict is: ", y_pred[0], label_names[int(round(y_pred[0,1]))])

plt.figure(figsize=(10,10))
for i in range(8):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    mappp = X_test[i]
    plt.imshow(mappp[:,:,0], cmap="inferno")
    plt.xlabel(label_names[int(round(y_pred[i,1]))])
plt.show()
