#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function

import numpy as np
import time

def CNN(inputs):

    input_shape = inputs["input_shape"]
    n_categories = inputs["n_categories"]
    kernel_size = inputs["kernel_size"]
    optimizer = inputs["optimizer"]
    loss = inputs["loss"]
    metrics = inputs["metrics"]
    input_activation = inputs["input_activation"]
    layer_activation = inputs["layer_activation"]
    output_activation = inputs["output_activation"]
    n_filters = inputs["n_filters"]
    n_nodes = inputs["n_nodes"]
    n_layers = inputs["n_layers"]
    batch_size = inputs["batch_size"]

    model = tf.keras.Sequential()

    model.add(layers.Conv2D(n_filters, kernel_size=kernel_size, activation=input_activation, input_shape=input_shape))
    print(model.output_shape)
    for i in range(n_layers):
        model.add(layers.Conv2D(n_nodes, kernel_size=kernel_size, activation=layer_activation))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
        print(model.output_shape)

    model.add(layers.Flatten())
    print(model.output_shape)
    model.add(layers.Dense(n_nodes, activation=layer_activation))
    print(model.output_shape)
    model.add(layers.Dense(n_categories, activation=output_activation))
    print(model.output_shape)
    print()

    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=10, verbose=1)


    sgd = tf.keras.optimizers.SGD(lr=0.01,momentum=0.95)

    if optimizer is None:
        model.compile(optimizer=sgd, loss=loss, metrics=metrics)
    else:
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

"""
tf.keras.layers.Conv2D(
    filters,
    kernel_size,
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    groups=1,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)


tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None, **kwargs
)


tf.keras.layers.Dense(
    units,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)



"""


"""
class NeuralNetwork:
    def __init__(self, inputs):
        self.inputs = inputs

    def get_inputs(self):
        # Should make it so number of activation functions is less difficult
        inputs = self.inputs

        kernel_size = inputs["kernel_s"]
        input_shape = inputs["input_s"]
        activ_1 = inputs["activ_1"]
        activ_2 = inputs["activ_2"]
        activ_3 = inputs["activ_3"]

        return kernel_size, input_shape, activ_1, activ_2, activ_3

    def create_model(self):
        inputs = self.inputs


"""
