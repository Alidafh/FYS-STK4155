#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from pathlib import Path
import tensorflow as tf

import numpy as np
import configuration as conf

def create_model():
    """
    creates a model using the configurations in the configuration file
    """

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(conf.n_filters,
                                    conf.kernel_size,
                                    activation=conf.input_activation,
                                    input_shape=conf.input_shape))

    model.add(tf.keras.layers.MaxPooling2D())
    for layer in conf.layer_config:
        model.add(tf.keras.layers.Conv2D(layer,
                                        kernel_size = conf.kernel_size,
                                        activation = conf.hidden_activation))

        model.add(tf.keras.layers.MaxPooling2D())
        model.add(tf.keras.layers.Dropout(0.15))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(conf.n_categories, activation=conf.output_activation))

    return model


def train_model(X_train, y_train, X_test, y_test, model, save_as=None, verbose=0):
    """
    train the model using the configurations in the configuration file
    """

    adam = tf.keras.optimizers.Adam(learning_rate=conf.learn_rate)
    sgd = tf.keras.optimizers.SGD(learning_rate=conf.learn_rate)

    if conf.optimizer == "adam":
        model.compile(optimizer=adam, loss=conf.loss, metrics=conf.metrics)

    if conf.optimizer == "sgd":
        model.compile(optimizer=sgd, loss=conf.loss, metrics=conf.metrics)

    if verbose == 1:
        print(model.summary())
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print("Untrained, accuracy: {:5.2f}%".format(100*acc),65*"_", sep="\n" )

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=conf.epochs)

    if save_as is not None:
        Path(conf.model_dir).mkdir(parents=True, exist_ok=True)
        model.save("{:}/model".format(save_as))

    return history



if __name__ == '__main__':
    import configuration as config
    create_model()

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
