#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

import configuration as conf
from CNN import create_model, train_model, create_model_3D, train_model_3D
from generate import load_data
from tools import gce, mnist

def plot_training_data(label_names=None, slice=0):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5, 5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(X_train[i][:,:,slice], cmap="gray")
        n = y_train[i].argmax()
        if label_names:
            plt.xlabel("{:}".format(label_names[n]))
        else:
            plt.xlabel("{:}".format(n))
    plt.show()

# GCE data
(X_train, y_train), (X_test, y_test) = gce(d3=True, seed=42, scale=True)
label_names = ["Clean", "DM"]

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_train[i][:,:,9,0], cmap="gray")
    n = y_train[i].argmax()
    plt.xlabel("{:}".format(label_names[n]))
plt.show()


model = create_model_3D()
history = train_model_3D(X_train, y_train, X_test, y_test, model, verbose=1, save_as="GCE_3D")
