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
    Creates a model using the configurations in the configuration file
    """

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(conf.n_filters,
                                    conf.kernel_size,
                                    activation=conf.input_activation,
                                    input_shape=conf.input_shape,
                                    kernel_regularizer=conf.reg,
                                    padding = "same"))

    model.add(tf.keras.layers.Conv2D(conf.n_filters,
                                    conf.kernel_size,
                                    activation=conf.hidden_activation,
                                    kernel_regularizer=conf.reg,
                                    padding="valid"))

    model.add(tf.keras.layers.MaxPooling2D())

    if conf.layer_config:
        for layer in conf.layer_config:
            model.add(tf.keras.layers.Conv2D(layer,
                                            kernel_size = conf.kernel_size,
                                            activation = conf.hidden_activation,
                                            kernel_regularizer=conf.reg,
                                            padding = "same"))

            model.add(tf.keras.layers.Conv2D(layer,
                                            kernel_size = conf.kernel_size,
                                            activation = conf.hidden_activation,
                                            kernel_regularizer=conf.reg,
                                            padding = "valid"))


            model.add(tf.keras.layers.MaxPooling2D())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(conf.connected_neurons, activation=conf.hidden_activation))
    model.add(tf.keras.layers.Dense(conf.n_categories, activation=conf.output_activation))

    return model


def train_model(X_train, y_train, X_val, y_val, model, save_as=None, verbose=0):
    """
    Train the model using the configurations in the configuration file
    """

    model.compile(optimizer=conf.opt, loss=conf.loss, metrics=conf.metrics)

    if verbose == 1:
        print(model.summary())
        loss, acc = model.evaluate(X_val, y_val, verbose=0)
        print("Untrained, accuracy: {:5.2f}%".format(100*acc),65*"_", sep="\n" )


    history = model.fit(X_train, y_train, batch_size = conf.batch_size,
                                          validation_data=(X_val, y_val),
                                          epochs=conf.epochs,
                                          callbacks = conf.early_stop)


    if save_as is not None:
        Path(conf.model_dir).mkdir(parents=True, exist_ok=True)
        model.save(conf.model_dir+"{:}".format(save_as))

    return history


def get_model(model_name):
    """ retrieve model from file """

    model = tf.keras.models.load_model(conf.model_dir+"{:}".format(model_name))

    return model



if __name__ == '__main__':
    create_model()
