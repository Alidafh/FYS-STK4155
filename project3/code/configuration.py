#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import tensorflow as tf

path = "../data/"
filename = "data_(1000, 28, 28, 20)_1.0_100.0_False_.npy"
data_file = path+filename
slice = None

###############################################################################
# for create_model()
###############################################################################
input_shape = (28, 28, 20)  # Shape of the images, holds the raw pixel values

n_filters = 16              # For the two first Conv2D layers
kernel_size = (3, 3)
layer_config = [32, 64]    # (layer1, layer2, layer3, ....)
connected_neurons = 128     # For the first Dense layer

n_categories = 2            # For the last Dense layer (2 for GCE, 10 for mnist)

input_activation  = "relu"
hidden_activation = "relu"
output_activation = "softmax"

reg = None          #tf.keras.regularizers.l2(l=0.1)

model_dir = "tmp/"           # Where to save the model

epochs = 10
batch_size = 10

#lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.1, decay_steps=1, decay_rate=1, staircase=False)
opt = tf.keras.optimizers.Adam(learning_rate=0.001)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

loss = "categorical_crossentropy" #loss = "mean_squared_error"
metrics = ["accuracy"]             #metrics = ["accuracy", tf.keras.metrics.AUC()]
