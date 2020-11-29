#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

import configuration as conf
from CNN import create_model, train_model
from generate import load_data

def plot_training_data(label_names=None, slice=0):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5, 5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(X_train[i][:,:,slice], cmap="inferno")
        if label_names:
            plt.xlabel("{:}".format(label_names[int(y_train[i])]))
        else:
            plt.xlabel("{:}".format(y_train[i]))
    plt.show()

def plot_test_results(label_names=None, slice=0):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(X_test[i][:,:,slice], cmap="inferno")
        n = y_pred[i].argmax()
        m = y_test[i].argmax()
        if label_names:
            l = "{:} (true={:})".format(label_names[n], label_names[m])
            plt.xlabel(l)
        else:
            plt.xlabel("{:} (true={:})".format(n,m))
    plt.show()


def plot_history(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    #plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

"""
# MNIST data

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

X_train, X_test = X_train / 255.0, X_test / 255.0
plot_training_data()

"""
# GCE data
maps, labels, stats = load_data(file=conf.data_file, slice=conf.slice)
label_names = ["Clean", "DM"]

X_train, X_test, y_train, y_test = train_test_split(maps, labels, test_size=0.2, random_state=42)

plot_training_data(label_names=label_names, slice=5)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = create_model()

#model.compile(optimizer="adam", loss=conf.loss, metrics=conf.metrics)
#print(model.summary())
#loss, acc = model.evaluate(X_test, y_test, verbose=0)
#print("Untrained, accuracy: {:5.2f}%".format(100*acc),65*"_", sep="\n" )

history = train_model(X_train, y_train, X_test, y_test, model, verbose=1)

loss, acc = model.evaluate(X_test, y_test, verbose=2)
print(65*"_", "accuracy: {:5.2f}%".format(100 * acc), 65*"_", sep="\n")

y_pred = model.predict(X_test)

plot_test_results(label_names=label_names, slice=5)
#plot_test_results()
plot_history(history)
