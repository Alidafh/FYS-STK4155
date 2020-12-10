#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tools import r2_score
from sklearn.model_selection import train_test_split


def create_model():
    """Creates a model using the configurations in the config file"""

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(conf.n_filters,
                                    conf.kernel_size,
                                    activation=conf.input_activation,
                                    input_shape=conf.input_shape,
                                    kernel_regularizer=conf.reg,
                                    bias_initializer="random_normal",
                                    padding = "same"))

    model.add(tf.keras.layers.MaxPooling2D())

    if conf.layer_config:
        for layer in conf.layer_config:
            model.add(tf.keras.layers.Conv2D(layer,
                                            kernel_size = conf.kernel_size,
                                            activation = conf.hidden_activation,
                                            kernel_regularizer=conf.reg,
                                            padding = "same"))


            model.add(tf.keras.layers.MaxPooling2D())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(conf.connected_neurons, activation=conf.hidden_activation))
    model.add(tf.keras.layers.Dense(conf.n_categories, activation=conf.output_activation))

    return model


def train_model(X_train, y_train, model, val_split = 0.2, save_as=None):
    """
    Train the model using the configurations in the configuration file
    A fraction val_split of the training data is used for validation
    """

    # Create path if it does not exist
    if save_as is not None:
        Path(conf.model_dir).mkdir(parents=True, exist_ok=True)


    # create validation data
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_split)


    # Compile the model and print summary
    model.compile(optimizer=conf.opt, loss=conf.loss, metrics=conf.metrics)
    print(model.summary())


    loss, metric = model.evaluate(X_val, y_val, verbose=0)
    if conf.type == "classification":
        print("Untrained, accuracy: {:5.2f}%".format(100*metric), 65*"_", sep="\n" )

    if conf.type == "regression":
        print("Untrained, r2_score: {:5.2f}%".format(100*metric), 65*"_", sep="\n")

    print("\ntraining:  ", X_train.shape[0])
    print("validation:", X_val.shape[0])
    print(65*"_")

    # set up history log
    csv_logger = None

    if save_as is not None:
        csv_logger = CSVLogger(conf.model_dir+save_as+'_training.log', separator=',', append=False)


    # Train the model
    model.fit(X_train, y_train, batch_size = conf.batch_size,
                                validation_split=0.2,
                                validation_data=(X_val, y_val),
                                epochs=conf.epochs,
                                callbacks = [csv_logger, conf.early_stop])


    if save_as is not None:
        name = conf.model_dir+save_as
        model.save(name)

    return model



def continue_training(X_train, y_train, model_name, val_split=0.2, save_as=None):
    """ continue training on existing model """

    # create validation data
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_split)

    print("\ntraining:  ", X_train.shape[0])
    print("validation:", X_val.shape[0])
    print(65*"_")

    m_name = conf.model_dir+"{:}".format(model_name)

    custom_objects = {"r2_score": r2_score} if conf.type=="regression" else None
    print(custom_objects)

    restored_model = tf.keras.models.load_model(m_name, custom_objects=custom_objects)

    np.testing.assert_allclose(
        restored_model.predict(X_val), restored_model.predict(X_val)
        )

    log = pd.read_csv(m_name+'_training.log', sep=',', engine='python')

    csv_logger = CSVLogger(m_name+'_training.log', separator=',', append=True)

    restored_model.fit(X_train, y_train, batch_size = conf.batch_size,
                                validation_data=(X_val, y_val),
                                epochs=conf.epochs,
                                callbacks = [csv_logger, conf.early_stop])

    if save_as is not None:
        restored_model.save(m_name)

    return restored_model

def arguments():
    import argparse
    import textwrap
    import sys

    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                          argparse.RawDescriptionHelpFormatter):
        pass

    description =textwrap.dedent("""\
    This is the Convolutional neural network used in FYS-STK 4155
    ---------------------------------------------------------------------------
    The configuration files that is used in this method are:\n
        - config_regression.py
        - config_classification.py.\n
    In these files you can change datafile, epochs, loss function etc. Indicate
    if you want to use the regression or classification option using the
    corresponding flags. We recommend that you supply a filename such that the
    CNN model is saved. This allows you to continue training at a later time,
    and simplifies further analysis. \n
    For more info: https://github.com/Alidafh/FYS-STK4155/tree/master/project3/code """)

    parser = argparse.ArgumentParser(description=description,
                                    formatter_class=CustomFormatter)

    required = parser.add_argument_group('required arguments')
    required.add_argument('-r', action='store_true', help="Regression")
    required.add_argument('-c', action='store_true', help="Classification")

    parser.add_argument('-n', type=str, metavar='name', action='store', default=None,
                    help='What name to store the model as')

    parser.add_argument('-e', action='store_true',
                    help='continue training on existing model, NOT DONE')

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
    """main script that trains the CNN etc."""

    X_train, X_test = conf.X_train, conf.X_test
    y_train, y_test = conf.y_train, conf.y_test


    if e:
        model = continue_training(X_train, y_train, val_split=0.2,
                                                    model_name=name,
                                                    save_as=name)

    else:
        model = create_model()
        model = train_model(X_train, y_train, val_split=0.2,
                                               model=model,
                                               save_as=name)


    loss, metric = model.evaluate(X_test, y_test, verbose=0)

    # print end result
    if conf.type == "classification":
        print(65*"_","\naccuracy: {:5.2f}%".format(100*metric),65*"_", sep="\n")

    if conf.type == "regression":
        print(65*"_","\nr2_score: {:5.2f}%".format(100*metric),65*"_", sep="\n")



if __name__ == '__main__':
    conf, name, e = arguments()
    main(conf, name, e)
