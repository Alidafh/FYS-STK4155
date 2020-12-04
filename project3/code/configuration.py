#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import tensorflow as tf

path = "../data/"
filename = "data_(10000, 28, 28, 10)_1_1_True_.npy"
data_file = path+filename
slice=None

"""
Configuration for the Convolutional neural network located in CNN.py
"""

###############################################################################
# For create_model_3D
###############################################################################
input_shape = (28, 28, 10, 1)

n_filters = 32              # For the two first Conv2D layers
kernel_size = (3, 3, 3)
#layer_config = (128, 128)   # (layer1, layer2, layer3, ....)
connected_neurons = 128     # For the first Dense layer
n_categories = 2            # For the last Dense layer (2 for GCE, 10 for mnist)

input_activation  = "relu"
hidden_activation = "relu"
output_activation = "softmax"

reg=None



model_dir = "tmp/"           # Where to save the model

epochs = 10
batch_size = 50

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

loss = "categorical_crossentropy"  #loss = "mean_squared_error"
metrics = ["accuracy"]          #metrics = ["accuracy", tf.keras.metrics.AUC()]


"""
###############################################################################
# for create_model()
###############################################################################
input_shape = (28, 28, 10)  # Shape of the images, holds the raw pixel values

n_filters = 32              # For the two first Conv2D layers
kernel_size = (3, 3)
layer_config = (128, 128)    # (layer1, layer2, layer3, ....)
connected_neurons = 256     # For the first Dense layer
n_categories = 2            # For the last Dense layer (2 for GCE, 10 for mnist)

#[32, 64, 128]
#[256, 512, 1024]
input_activation  = "relu"
hidden_activation = "relu"
output_activation = "softmax"

reg = None

model_dir = "tmp/"           # Where to save the model

epochs = 10
batch_size = 50


lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.01, decay_steps=1, decay_rate=1, staircase=False)
opt = tf.keras.optimizers.Adam(learning_rate=0.001)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

loss = "categorical_crossentropy"  #loss = "mean_squared_error"
metrics = ["accuracy"]          #metrics = ["accuracy", tf.keras.metrics.AUC()]



"""

















"""
path = "../data/"
filename = "data_(2000, 28, 28, 10)_1_1_True_.npy"
data_file = path+filename
slice=None


input_shape = (28, 28, 10)  # Shape of the images, holds the raw pixel values

n_filters = 64              # For the two first Conv2D layers
kernel_size = (3, 3)
layer_config = (128, 128)     # (layer1, layer2, layer3, ....)
connected_neurons = 512     # For the first Dense layer
n_categories = 2            # For the last Dense layer (2 for GCE, 10 for mnist)

input_activation  = "relu"
hidden_activation = "relu"
output_activation = "softmax"

#reg = tf.keras.regularizers.l2(l=0.001)
reg = None


model_dir = "tmp/"           # Where to save the model

epochs = 10
batch_size = 10

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.01, decay_steps=10, decay_rate=6, staircase=False)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
#opt = tf.keras.optimizers.Adam(learning_rate=0.001)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

loss = "mean_squared_error"     #loss = "categorical_crossentropy"
metrics = ["accuracy"]          #metrics = ["accuracy", tf.keras.metrics.AUC()]

"""
