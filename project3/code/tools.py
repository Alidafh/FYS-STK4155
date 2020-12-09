#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
from generate import load_data
from sklearn.model_selection import train_test_split


def preprocess(maps, labels, train_size, strength=False, scale=True, seed=None):
    """
    Scale data and plit into training and test sets
    """
    if strength == False: labels = to_categorical(labels)
    if scale == True: maps = maps/maps.max()

    X_train, X_test, y_train, y_test = train_test_split(maps, labels,
                                                        train_size=train_size,
                                                        random_state=seed)
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    return (X_train, y_train), (X_test, y_test)



def coeff_determination(y_true, y_pred):
    """
    Use R2 score as a measure of how good the model works
    https://jmlb.github.io/ml/2017/03/20/CoeffDetermination_CustomMetric4Keras/
    """
    SS_res =  K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))









def mnist():
    # MNIST data
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.reshape(60000,28,28,1)
    X_test = X_test.reshape(10000,28,28,1)

    X_train = X_train/255.
    X_test = X_test/255.

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return (X_train, y_train), (X_test, y_test)


def plot_training_data(X_train, y_train, label_names=None, slice=0):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5, 5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(X_train[i][:,:,slice], cmap="inferno")
        if label_names:
            n = y_train[i].argmax()
            plt.xlabel("{:}".format(label_names[n]))
        else:
            y_train = y_train.ravel()
            n = y_train[i]
            plt.xlabel("{:.4f}".format(n))
    plt.show()

def plot_test_results(y_test, y_pred, label_names=None, slice=0):
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
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.show()
    plt.show()
