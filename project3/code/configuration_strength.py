#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 21:03:05 2020

@author: gert
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import tensorflow as tf

path = "../data/"

filename = "maps_(1500, 28, 28, 20)_100.0_True_.npy"

data_file = path+filename
slice = None

###############################################################################
# for create_model()
###############################################################################
input_shape = (28, 28, 20)  # Shape of the images, holds the raw pixel values



n_filters = 16              # For the two first Conv2D layers
kernel_size = (5,5)
layer_config = (8, 4)    # (layer1, layer2, layer3, ....)
connected_neurons = 32     # For the first Dense layer

n_categories = 1            # For the last Dense layer (2 for GCE, 10 for mnist)

input_activation  = "relu"
hidden_activation = "relu"
output_activation = "softmax"

reg = None #tf.keras.regularizers.l2(l=0.1)

model_dir = "tmp/"           # Where to save the model


epochs = 50
batch_size = 50

#lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.1, decay_steps=1, decay_rate=1, staircase=False)
opt = tf.keras.optimizers.SGD(learning_rate=0.01)


lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.01, decay_steps=1, decay_rate=1, staircase=False)
opt = tf.keras.optimizers.Adam(learning_rate=0.00001)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)


loss = "mean_squared_error"  #loss = "mean_squared_error"
metrics = ["mean_squared_error"]          #metrics = ["accuracy", tf.keras.metrics.AUC()]






"""
###############################################################################
# For create_model_3D
###############################################################################
input_shape = (28, 28, 10, 1)

n_filters = 32              # For the two first Conv2D layers
kernel_size = (5, 5, 5)
connected_neurons = 128     # For the first Dense layer
=======
n_filters = 16              # For the two first Conv2D layers
kernel_size = (3, 3)
layer_config = [32, 64]    # (layer1, layer2, layer3, ....)
connected_neurons = 128     # For the first Dense layer

>>>>>>> origin/master
n_categories = 2            # For the last Dense layer (2 for GCE, 10 for mnist)

input_activation  = "relu"
hidden_activation = "relu"
output_activation = "softmax"

<<<<<<< HEAD
reg=None

model_dir = "tmp/"           # Where to save the model

epochs = 5
batch_size = 100

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

loss = "mean_squared_error"   #loss = "categorical_crossentropy"  #
metrics = ["accuracy"]          #metrics = ["accuracy", tf.keras.metrics.AUC()]


"""


# u = 0
# plt.figure(u)
# plt.imshow(X_train[u][:,:,10])

