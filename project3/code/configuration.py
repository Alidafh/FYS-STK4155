#!/usr/bin/env python3
# -*- coding: utf-8 -*-

path = "../data/"
filename = "data_(200, 50, 50, 10)_1_0.1_True_.csv"
data_file = path+filename

"""
Configuration for the Convolutional neural network located in CNN.py
"""
"""
input_shape = (50, 50, 1)   # Shape of images

n_filters = 64
n_categories = 2
kernel_size = 3
layer_config = (32, 32)     # (layer1, layer2, layer3 ....)

input_activation  = "relu"
hidden_activation = "relu"
output_activation = "softmax"


model_dir = "tmp"

optimizer = "adam"
learn_rate = 0.001
loss = "categorical_crossentropy"
metrics = ["accuracy"]
epochs = 10



"""
input_shape = (28, 28, 1)

n_filters = 64
n_categories = 10
kernel_size = 3
layer_config = (32, 32)

input_activation  = "relu"
hidden_activation = "relu"
output_activation = "softmax"


# for fit_model():
model_dit = "tmp"

optimizer = "adam"
learn_rate = 0.001

loss = "categorical_crossentropy"
metrics = ["accuracy"]

epochs = 5
