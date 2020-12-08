#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 11:31:19 2020

@author: gert
"""

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

import configuration_mnist as conf
from CNN_mnist import create_model_3D, create_model_2D, train_model
from generate import load_data

import mnist_loader



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

def mnist():
    # MNIST data
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.reshape(60000,28,28,1)
    X_test = X_test.reshape(10000,28,28,1)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    plot_training_data()
    return (X_train, y_train), (X_test, y_test)


# # GCE data


# maps, labels, stats = load_data(file=conf.data_file, slice=conf.slice)

# label_names = ["Clean", "DM"]
# labels = to_categorical(labels)

# maps = maps/maps.max()

# X_train, X_test, y_train, y_test = train_test_split(maps, labels, train_size=0.8)


training_data, validation_data, test_data =  mnist_loader.load_data_wrapper()

training_data = list(training_data)
test_data = list(test_data)

X_train = []
y_train = []
X_test = []
y_test = []

for i in training_data:
    X_train.append(i[0])
    y_train.append(i[1])
    
for j in test_data:
    X_test.append(j[0])
    y_test.append(j[1])

X_train = np.array(X_train).reshape(len(training_data),28,28,1)
y_train = np.array(y_train).reshape(len(training_data),10)
X_test = np.array(X_test).reshape(len(test_data), 28,28, 1)
y_test = to_categorical(np.array(y_test)).reshape(len(test_data), 10)


# X_train = X_train[:,:,:,:, np.newaxis]
# X_test = X_test[:,:,:,:, np.newaxis]


#plot_training_data(label_names=label_names, slice=9)

model = create_model_2D()
history = train_model(X_train, y_train, X_test, y_test, model, verbose=1, save_as="GCE")

loss, acc = model.evaluate(X_test, y_test, verbose=2)
print(65*"_", "accuracy: {:5.2f}%".format(100 * acc), 65*"_", sep="\n")

y_pred = model.predict(X_test)

#plot_test_results(label_names=label_names)
#plot_test_results()
plot_history(history)

