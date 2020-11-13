#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:47:18 2020

@author: gert
"""
import regression2 as reg
import numpy as np
import mnist_loader

training_data, validation_data, test_data =  mnist_loader.load_data_wrapper()

training_data = list(training_data)
validation_data = list(validation_data)
test_data = list(test_data)

print("before split:")
print(type(training_data), np.shape(training_data))
print()

x, y = reg.split_data(training_data)

print("after split:")
print("x:", type(x),np.shape(x))
print("y:", type(y), np.shape(y))
print()

np.random.seed(42)

net = reg.Network([784, 30, 10])

# net.regularization_type = reg.L2

act = reg.Softmax()

# net.afs[-1] = act

net.standard_af = act
net.set_afs_to_standard()

# net.biases = [np.random.randn(y,1) + 1 for y in net.layer_sizes[1:]]  # First layer is input, thus no biases.
# net.weights = [np.random.randn(y,x) + 1 for x, y in zip(net.layer_sizes[:-1], net.layer_sizes[1:])]

net.train(x, y, n_epochs = 1, learn_rate=3, test_data=test_data, classification=True, lamb=5)


"""
alida ~/Documents/uio/Master/FYS-STK4155-1/project2/gert # $ python main5.py
before split:
<class 'list'> (50000, 2)

after split:
x: <class 'list'> (50000, 784, 1)
y: <class 'list'> (50000, 10, 1)

NN, train (classification):  True
Regression, SGD(0)
Regression, SGD(1)
NN, gradient_data_point (X, y) (784, 1) (10, 1)
NN, gradient_data_point (beta_temp) (4,)
NN, gradient_data_point (weights_temp) (2,)
NN, gradient_data_point (biases_temp) (2,)
NN, gradient_data_point (nabla_b) (2,)
NN, gradient_data_point (nabla_w) (2,)

<class 'numpy.ndarray'> (30, 1)
<class 'numpy.ndarray'> (30, 784)
<class 'numpy.ndarray'> (10, 1)
<class 'numpy.ndarray'> (10, 30)
"""
"""
alida ~/Documents/uio/Master/FYS-STK4155-1/project2/gert # $ python main6.py
NN, train (classification):  False
Regression, SGD(0)
Regression, SGD(1)
NN, gradient_data_point (X, y) (28,) ()
NN, gradient_data_point (beta_temp) (4,)
NN, gradient_data_point (weights_temp) (2,)
NN, gradient_data_point (biases_temp) (2,)
NN, gradient_data_point (nabla_b) (2,)
NN, gradient_data_point (nabla_w) (2,)

<class 'numpy.ndarray'> (30, 1)
<class 'numpy.ndarray'> (30, 28)
<class 'numpy.ndarray'> (100, 1)
<class 'numpy.ndarray'> (100, 30)

"""
