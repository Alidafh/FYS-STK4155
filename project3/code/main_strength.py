#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 21:03:42 2020

@author: gert
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

import configuration_strength as conf

from CNN_strength import create_model_3D, create_model, train_model

from generate import load_data
from tools_strength import gce_strength, mnist, plot_training_data, plot_test_results, plot_history


label_names = ["Clean", "DM"]

(X_train, y_train), (X_test, y_test) = gce_strength(seed=42, scale=True)

model = create_model()

history = train_model(X_train, y_train, X_test, y_test, model, verbose=1, save_as="GCE")

loss, acc = model.evaluate(X_test, y_test, verbose=2)
print(65*"_", "accuracy: {:5.2f}%".format(100 * acc), 65*"_", sep="\n")

#plot_test_results(label_names=label_names)
#plot_test_results()
plot_history(history)
