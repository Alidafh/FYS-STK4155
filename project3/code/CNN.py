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


def train_model(X_train, y_train, model, val_split = 0.2, verbosity=1, save_as=None):
    """
    Train the model using the configurations in the configuration file
    A fraction val_split of the training data is used for validation
    """

    # Create path if it does not exist
    if save_as is not None:
        Path(conf.model_dir).mkdir(parents=True, exist_ok=True)


    # create validation data
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_split)


    # Compile the model
    model.compile(optimizer=conf.opt, loss=conf.loss, metrics=conf.metrics)

    if verbosity != 0:
        model.summary()
        loss, metric = model.evaluate(X_val, y_val, verbose=0)
        print(f"\ntraining: {X_train.shape[0]} - validation: {X_val.shape[0]} - Untrained, {model.metrics_names[1]}: {100*metric:.2f}%")
        print(65*"_", "\n")


    # set up history log
    if save_as is not None:
        csv_logger = CSVLogger(conf.model_dir+save_as+'_training.log', separator=',', append=False)
        callbacks = [csv_logger, conf.early_stop, conf.reduce_lr]

    else:
        callbacks = [conf.early_stop, conf.reduce_lr]


    # Train the model
    model.fit(X_train, y_train, batch_size = conf.batch_size,
                                validation_split=0.2,
                                validation_data=(X_val, y_val),
                                epochs=conf.epochs,
                                callbacks = callbacks,
                                verbose=verbosity)


    if save_as is not None:
        name = conf.model_dir+save_as
        model.save(name)

    return model



def continue_training(X_train, y_train, model_name, val_split=0.2, save_as=None):
    """ continue training on existing model """

    # create validation data
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_split)

    # name of the model
    m_name = conf.model_dir+"{:}".format(model_name)

    # If the type is regression we need to specify to retrieve the r2_score
    custom_objects = {"r2_score": r2_score} if conf.type=="regression" else None

    restored_model = tf.keras.models.load_model(m_name, custom_objects=custom_objects)

    # Check if it worked
    np.testing.assert_allclose(restored_model.predict(X_val),
                               restored_model.predict(X_val) )


    # Continues in the same log as before
    csv_logger = CSVLogger(m_name+'_training.log', separator=',', append=True)

    restored_model.fit(X_train, y_train, batch_size = conf.batch_size,
                                validation_data=(X_val, y_val),
                                epochs=conf.epochs,
                                callbacks = [csv_logger, conf.early_stop])

    # Save the model again
    if save_as is not None:
        restored_model.save(m_name)

    return restored_model



def cross_validate(X, y, num_folds=2, verbosity=0, save_as=None):
    """K-fold Cross Validation model evaluation"""

    from sklearn.model_selection import KFold


    if save_as is not None:
        kfold_path="/kFold_"+conf.type+"/"
        Path(conf.model_dir+kfold_path).mkdir(parents=True, exist_ok=True)


    metric_per_fold = []
    loss_per_fold = []

    fold_no = 1
    no_classes = 1
    kf = KFold(n_splits=num_folds, shuffle=True)

    for train_index, test_index in kf.split(X,y):
        model = create_model()

        if verbosity==0:
            print(f"Training for fold {fold_no}...", end='\r', flush=True)


        name_tmp = kfold_path+save_as+"_"+str(fold_no)

        model = train_model(X[train_index], y[train_index], val_split=0.2,
                                                            model=model,
                                                            verbosity=verbosity,
                                                            save_as=name_tmp)

        # Generate generalization metrics
        scores = model.evaluate(X[test_index], y[test_index], verbose=0)

        result1 = f" - {model.metrics_names[0]}: {scores[0]:.4f}"
        result2 = f" - {model.metrics_names[1]}: {scores[1]:.4f}"

        if verbosity==0:
            print(f"Training for fold {fold_no}"+result1+result2)


        loss_per_fold.append(scores[0])
        metric_per_fold.append(scores[1])

        fold_no = fold_no + 1   # Increase fold number


    avg_loss = np.mean(loss_per_fold)
    std_loss = np.std(loss_per_fold)

    avg_metric = np.mean(metric_per_fold)
    std_metric = np.std(metric_per_fold)

    print(65*"_", "\n")
    print(f"avg. loss:     {avg_loss:.4f} (+- {std_loss:.4f})")
    print(f"avg. {model.metrics_names[1]}: {avg_metric:.4f} (+- {std_metric:.4f})")



def main(conf, name, resume, validate):
    """main script that trains the CNN etc."""

    # easier to not have to write conf. on these
    X_train, X_test = conf.X_train, conf.X_test
    y_train, y_test = conf.y_train, conf.y_test


    if validate:
        print(f"\nPerforming {validate}-Fold cross validation\n"+ 64*"_"+"\n")
        print(f"Analysis: {conf.type}\n"+64*"_"+"\n")

        # Merge inputs and targets
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)

        cross_validate(X, y, num_folds=validate, verbosity=0, save_as=name)



    elif resume:
        print(f"\nResuming training on: {name}\n"+64*"_"+"\n")
        print(f"Analysis: {conf.type}\nSave as:  {name}\n"+64*"_"+"\n")

        model = continue_training(X_train, y_train, val_split=0.2,
                                                    model_name=name,
                                                    save_as=name)

        loss, metric = model.evaluate(X_test, y_test, verbose=0)
        print(65*"_", f"\n{model.metrics_names[1]}: {100*metric:.2f}%", 65*"_", sep="\n")


    else:
        print(64*"_"+"\n"+f"\nAnalysis: {conf.type}\nSave as:  {name}\n"+64*"_"+"\n")

        model = create_model()
        model = train_model(X_train, y_train, val_split=0.2,
                                               model=model,
                                               save_as=name)

        loss, metric = model.evaluate(X_test, y_test, verbose=0)
        print(65*"_", f"\n{model.metrics_names[1]}: {100*metric:.2f}%", 65*"_", sep="\n")





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

    parser.add_argument('-n', type=str, metavar='name', default=None,
                        help='What name to store the model as')

    parser.add_argument('-e', type=str, metavar="name", nargs='?', default=None, const=True,
                        help='Continue training on given model')

    parser.add_argument('-v', type=int, nargs='?', default=None, const=5,
                        help='kFold validation, cannot be used with e')

    args = parser.parse_args()

    r, c, n, e, v = args.r, args.c, args.n, args.e, args.v


    if not r and not c or r and c:
        parser.print_help()
        sys.exit("\nUsage error: You need to choose ONE of the required arguments.")


    if v and e:
        parser.print_help()
        sys.exit("\nUsage error: You cannot use -v with -e ")


    if e:
        if isinstance(e, str) and not n:
            n = e

        if (not isinstance(e, str) and not n):
            parser.print_help()
            sys.exit("\nUsage error: You need to specify the name of the model you want to resume training on")

        if isinstance(e, str) and n and e != n:
            parser.print_help()
            sys.exit("\nUsage error: Two different model names were given, can only train on one")



    if r: import config_regression as conf
    if c: import config_classification as conf

    return conf, n, e, v


if __name__ == '__main__':
    conf, name, resume, validate = arguments()
    main(conf, name, resume, validate)
