#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration file for the classification CNN
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import tensorflow as tf
from generate import load_data
from tools import preprocess

###############################################################################
# Set up the data
###############################################################################
type = "classification"
path = "../data/"
filename = "data_(2000, 28, 28, 20)_0.5_100.0_True_.npy"
data_file = path+filename
slice = None

maps, labels, stats = load_data(file=data_file, slice=slice)
(X_train, y_train), (X_test, y_test) = preprocess(maps, labels,
                                                train_size = 0.8,
                                                strength=False,
                                                scale=True,
                                                seed=42)

label_names = ["clean", "dm"]
###############################################################################
# for create_model()
###############################################################################
input_shape = (28, 28, 20)  # Shape of the images, holds the raw pixel values

n_filters = 16              # For the two first Conv2D layers
kernel_size = (5, 5)
layer_config = [32, 64]     # [layer1, layer2, layer3, ....] or None for no hidden layers
connected_neurons = 128     # For the first Dense layer

n_categories = 2            # For the last Dense layer (2 for GCE, 10 for mnist)

input_activation  = "relu"
hidden_activation = "relu"
output_activation = "softmax"

reg = None #tf.keras.regularizers.l2(l=0.001)

###############################################################################
# for train_model()
###############################################################################
model_dir = "tmp/"           # Where to save the model

epochs = 20
batch_size = 10

#lr3 = tf.keras.optimizers.schedules.InverseTimeDecay(1e-2, decay_steps=2, decay_rate=10, staircase=True)
opt = tf.keras.optimizers.Adam(learning_rate=1e-5)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

loss = "categorical_crossentropy"  #loss = "mean_squared_error"
metrics = ["accuracy"]          #metrics = ["accuracy", tf.keras.metrics.AUC()]
