#!/usr/bin/env python3
# -*- coding: utf-8 -*-

path = "../data/"
filename = "data_(4000, 28, 28, 10)_1_0.1_True_.csv"
data_file = path+filename
slice=None

"""
Configuration for the Convolutional neural network located in CNN.py
"""

input_shape = (28, 28, 10)   # Shape of the images, holds the raw pixel values

n_filters = 64
n_categories = 2            # Number of categories, 2 for GCE and 10 for mnist
kernel_size = 3
layer_config = (128, 128)   # (layer1, layer2, layer3 ....)

input_activation  = "relu"
hidden_activation = "relu"
#output_activation = "softmax"
output_activation = "sigmoid"


model_dir = "tmp"

optimizer = "adam"
learn_rate = 0.00001
loss = "categorical_crossentropy"
metrics = ["accuracy"]
epochs = 5
