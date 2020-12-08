#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 21:16:08 2020

@author: gert
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from pathlib import Path
import tensorflow as tf
import numpy as np
import configuration_strength as conf

def create_model():
    """
    creates a model using the configurations in the configuration file
    """

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(conf.n_filters,
                                    conf.kernel_size,
                                    padding = "same",
                                    activation=conf.input_activation,
                                    input_shape=conf.input_shape,
                                    kernel_regularizer=conf.reg))

    model.add(tf.keras.layers.Conv2D(conf.n_filters, conf.kernel_size,
                                    activation=conf.hidden_activation,
                                    kernel_regularizer=conf.reg,
                                    padding="valid"))


    model.add(tf.keras.layers.MaxPooling2D())



    # for layer in conf.layer_config:
    #     model.add(tf.keras.layers.Conv2D(layer,
    #                                     kernel_size = conf.kernel_size,
    #                                     activation = conf.hidden_activation,
    #                                     kernel_regularizer=conf.reg,
    #                                     padding = "same"))

    #     model.add(tf.keras.layers.Conv2D(layer,
    #                                     kernel_size = conf.kernel_size,
    #                                     activation = conf.hidden_activation,
    #                                     kernel_regularizer=conf.reg,
    #                                     padding = "valid"))




    #     model.add(tf.keras.layers.MaxPooling2D())
        


    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(conf.connected_neurons, activation=conf.hidden_activation))
    model.add(tf.keras.layers.Dense(conf.n_categories, activation=conf.output_activation))

    return model


def create_model_3D():
    """
    creates a model using the configurations in the configuration file
    """

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv3D(conf.n_filters,
                                    conf.kernel_size,
                                    padding = "same",
                                    activation=conf.input_activation,
                                    input_shape=conf.input_shape,
                                    kernel_regularizer=conf.reg))

    model.add(tf.keras.layers.Conv3D(conf.n_filters, conf.kernel_size,
                                    activation=conf.input_activation,
                                    kernel_regularizer=conf.reg,
                                    padding = "valid"))


    model.add(tf.keras.layers.MaxPooling3D(padding="same"))

    for layer in conf.layer_config:
        model.add(tf.keras.layers.Conv3D(layer,
                                        kernel_size = conf.kernel_size,
                                        activation = conf.hidden_activation,
                                        kernel_regularizer=conf.reg,
                                        padding = "same"))

        model.add(tf.keras.layers.Conv3D(layer,
                                        kernel_size = conf.kernel_size,
                                        activation = conf.hidden_activation,
                                        kernel_regularizer=conf.reg,
                                        padding = "valid"))

        model.add(tf.keras.layers.MaxPooling3D(padding="same"))

        #model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(conf.connected_neurons, activation=conf.hidden_activation))
    model.add(tf.keras.layers.Dense(conf.n_categories, activation=conf.output_activation))

    return model




def train_model_3D(X_train, y_train, X_val, y_val, model, save_as=None, verbose=0):

    model.compile(optimizer=conf.opt, loss=conf.loss, metrics=conf.metrics)

    if verbose == 1:
        print(model.summary())
        loss, acc = model.evaluate(X_val, y_val, verbose=0)
        print("Untrained, accuracy: {:5.2f}%".format(100*acc),65*"_", sep="\n" )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    # fit the model
    history = model.fit(X_train, y_train, batch_size = conf.batch_size,
                                          validation_data=(X_val, y_val),
                                          epochs=conf.epochs,
                                          callbacks = conf.early_stop)

    if save_as is not None:
        # Save the model for later use
        Path(conf.model_dir).mkdir(parents=True, exist_ok=True)
        model.save(conf.model_dir+"{:}".format(save_as))

    return history


def train_model(X_train, y_train, X_val, y_val, model, save_as=None, verbose=0):
    """
    Train the model using the configurations in the configuration file
    """

    # Compile the model
    model.compile(optimizer=conf.opt, loss=conf.loss, metrics=conf.metrics)

    if verbose == 1:
        print(model.summary())
        loss, acc = model.evaluate(X_val, y_val, verbose=0)
        print("Untrained, accuracy: {:5.2f}%".format(100*acc),65*"_", sep="\n" )


    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    # fit the model
    history = model.fit(X_train, y_train, batch_size = conf.batch_size,
                                          validation_data=(X_val, y_val),
                                          epochs=conf.epochs,
                                          callbacks = conf.early_stop)

    if save_as is not None:
        # Save the model for later use
        Path(conf.model_dir).mkdir(parents=True, exist_ok=True)
        model.save(conf.model_dir+"{:}".format(save_as))

    return history


def get_model(model_name):
    """ retrieve model from file """
    model = tf.keras.models.load_model(conf.model_dir+"{:}".format(model_name))
    return model



if __name__ == '__main__':
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
