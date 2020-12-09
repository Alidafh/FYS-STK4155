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
from tools import preprocess, coeff_determination

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
                                                strength=True,
                                                scale=True,
                                                seed=42)

###############################################################################
# for create_model()
###############################################################################

input_shape = (28, 28, 20)  # Shape of the images, holds the raw pixel values

n_filters = 16              # For the two first Conv2D layers
kernel_size = (5,5)
layer_config = None#[32]         # (layer1, layer2, layer3, ....)

connected_neurons = 64      # For the first Dense layer
n_categories = 1            # For the last Dense layer

input_activation  = "relu"
hidden_activation = "relu"
output_activation = "sigmoid"

reg = None  #tf.keras.regularizers.l2(l=0.1)

###############################################################################
# for train_model()
###############################################################################

model_dir = "tmp/"           # Where to save the model

epochs = 10
batch_size = 50

#lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.01, decay_steps=1, decay_rate=1, staircase=False)
opt = tf.keras.optimizers.Adam(learning_rate=1e-5)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

loss = "mean_squared_error"

#metrics = [coeff_determination]
metrics = ["mean_squared_error"]
