#!/usr/bin/env python3
# -*- coding: utf-8 -*-

path = "../data/"
filename = "data_(8000, 28, 28, 10)_1_1_True_.csv"
data_file = path+filename
slice=None

"""
Configuration for the Convolutional neural network located in CNN.py
"""

input_shape = (28, 28, 10)   # Shape of the images, holds the raw pixel values

n_filters = 32
n_categories = 2            # Number of categories, 2 for GCE and 10 for mnist
kernel_size = 3
layer_config = (64, 128)   # (layer1, layer2, layer3 ....)


input_activation  = "relu"
hidden_activation = "softmax"
output_activation = "sigmoid"


model_dir = "tmp"

optimizer = "adam"          # "sgd"
learn_rate = 0.001
loss = "categorical_crossentropy"
metrics = ["accuracy"]
epochs = 5
