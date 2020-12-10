#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
NOT DONE JUST AN EXAMPLE
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tools import r2_score, history_classification
import config_classification as conf
import pandas as pd

# easier to use without writing conf all the time
X_train, X_test = conf.X_train, conf.X_test
y_train, y_test = conf.y_train, conf.y_test

# get the saved model
model_name = conf.model_dir+"clas"
model = tf.keras.models.load_model(model_name)

# print summary
model.summary()

# print the loss and r2 score
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("loss: %.4f" %loss, "accuracy:  %.4f" %acc, sep='\n')
print()

# get the log file
log_data = pd.read_csv(model_name+'_training.log', sep=',', engine='python')

# plot the loss and r2 vs number of epochs
history_classification(log_data, title="reg1_history")

plt.show()
quit()
plt.figure(figsize=(10,10))
slice=0
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_test[i][:,:,slice], cmap="inferno")
    n = y_pred[i].argmax()
    m = y_test[i].argmax()
    #print("true:", y_test[i], "\t predicted:", np.array2string(y_pred[i], precision=2))
    l = "{:} (true={:})".format(conf.label_names[n], conf.label_names[m])
    plt.xlabel(l)
plt.show()

log_data = pd.read_csv(model_name+'_training.log', sep=',', engine='python')
