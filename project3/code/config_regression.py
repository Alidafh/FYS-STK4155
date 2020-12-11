#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configureation file for the regression CNN
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import tensorflow as tf
from generate import load_data
from tools import preprocess, r2_score

###############################################################################
# Set up the data
###############################################################################

type = "regression"

path = "../data/"
filename = "maps_(1500, 28, 28, 20)_100.0_True_.npy"
data_file = path+filename
slice = None

maps, labels, stats = load_data(file=data_file, slice=slice)
(X_train, y_train), (X_test, y_test) = preprocess(maps, labels,
                                                train_size = 0.8,
                                                regress=True,
                                                scale=True,
                                                seed=42,
                                                shuffle=True)

###############################################################################
# for create_model()
###############################################################################

input_shape = (28, 28, 20)  # Shape of the images, holds the raw pixel values

n_filters = 16              # For the two first Conv2D layers
kernel_size = (5,5)
layer_config = [32, 64]         # (layer1, layer2, layer3, ....)

connected_neurons = 128      # For the first Dense layer
n_categories = 1            # For the last Dense layer

input_activation  = "relu"
hidden_activation = "relu"
output_activation = "sigmoid"

reg = None  #tf.keras.regularizers.l2(l=0.1)

###############################################################################
# for train_model()
###############################################################################

model_dir = "tmp/"           # Where to save the model

epochs = 100
batch_size = 10

opt = tf.keras.optimizers.Adam(learning_rate=1e-5)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

loss = "mean_squared_error"
metrics = [r2_score]
