#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from pathlib import Path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
    from tensorflow.keras.callbacks import CSVLogger
    model.compile(optimizer=conf.opt, loss=conf.loss, metrics=conf.metrics)

    if verbose != 0:
        print(model.summary())

        if conf.type == "classification":
            loss, acc = model.evaluate(X_val, y_val, verbose=0)
            print("Untrained, accuracy: {:5.2f}%".format(100*acc),65*"_", sep="\n" )

        if conf.type == "regression":
            loss, r2 = model.evaluate(X_val, y_val, verbose=0)
            print(65*"_", "Untrained, r2: {:5.2f}%".format(100*r2), 65*"_", sep="\n")


    csv_logger = None

    if save_as is not None:
        Path(conf.model_dir).mkdir(parents=True, exist_ok=True)
        csv_logger = CSVLogger(conf.model_dir+save_as+'_training.log', separator=',', append=False)

    history = model.fit(X_train, y_train, batch_size = conf.batch_size,
                                          validation_data=(X_val, y_val),
                                          epochs=conf.epochs,
                                          callbacks = [csv_logger, conf.early_stop])

    if save_as is not None:
        model.save(conf.model_dir+"{:}".format(save_as))

    return history


def continue_training(X_train, y_train, X_val, y_val, model_name, save_as=None, verbose=0):
    """ continue training on existing model NOT DONE """

    model = tf.keras.models.load_model(conf.model_dir+"{:}".format(model_name))

    np.testing.assert_allclose(
        model.predict(test_input), reconstructed_model.predict(test_input)
        )

    history = model.fit(X_train, y_train, batch_size = conf.batch_size,
                                          validation_data=(X_val, y_val),
                                          epochs=conf.epochs,
                                          callbacks = conf.early_stop)


    if save_as is not None:
        Path(conf.model_dir).mkdir(parents=True, exist_ok=True)
        model.save(conf.model_dir+"{:}".format(save_as))

    return history

def arguments():
    """
    Choose which type of network you want to run using command line flags.

    """
    import argparse
    import sys

    description =  """Train the CNN"""
    frm =argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=frm)

    required = parser.add_argument_group('required arguments')
    required.add_argument('-r', action='store_true', help="Regression")
    required.add_argument('-c', action='store_true', help="Classification")

    parser.add_argument('-n', type=str, metavar='name', action='store', default=None,
                    help='What name to store the model as')

    parser.add_argument('-e', action='store_true', help='continue training on existing model, NOT DONE')

    args = parser.parse_args()

    r, c, name, e = args.r, args.c, args.n, args.e

    if not r and not c or r and c:
        parser.print_help()
        print("\nUsage error: you need to choose ONE of the required arguments.")
        sys.exit(1)

    print(64*"_", "\n")
    if e:
        print("resuming training..")

    if r:
        import config_regression as conf
        #print(64*"_", "\n")
        print("Analysis:   Regression")
        print("Model name: {:}".format(name))
        print(64*"_")


    if c:
        import config_classification as conf
        #print(64*"_", "\n")
        print("Analysis:   Classification")
        print("Model name: {:}".format(name))
        print(64*"_")

    return conf, name, e

def main(conf, name, e):
    """
    main script that trains the CNN etc.
    """

    X_train, X_test, y_train, y_test = conf.X_train, conf.X_test, conf.y_train, conf.y_test

    if e:
        history = continue_training(X_train, y_train, X_test, y_test, name, save_as=name, verbose=1)
    else:
        model = create_model()
        history = train_model(X_train, y_train, X_test, y_test,
                        model=model, verbose=1, save_as=name)


    if conf.type == "classification":
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(65*"_", "accuracy: {:5.2f}%".format(100 * acc), 65*"_", sep="\n")


    if conf.type == "regression":
        loss, r2 = model.evaluate(X_test, y_test, verbose=0)
        print(65*"_", "R2: {:5.2f}%".format(100*r2), 65*"_", sep="\n")





if __name__ == '__main__':
    conf, name, e = arguments()
    main(conf, name, e)
