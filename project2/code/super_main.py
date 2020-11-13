#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 14:21:45 2020

@author: gert
"""


import skynet as sky
import mnist_loader
import tools
import quickplot as qupl

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split




###############################################################################
################## Handily available plotting parameters ######################
###############################################################################

qp = qupl.QuickPlot()

qp.label_size = 30                     # Size of title and x- and y- labels

qp.line_width = 3                      # Thickness of plotted lines

qp.label_pad = 17                      # Distance between axis label and axis

qp.tick_size = 22                      # Size of axis ticks

qp.legend_size = 22                    # Size of legend



###############################################################################
#################### Load and prepare Regression Data #########################
###############################################################################

# Input parameters:
    
ndata_franke = 10000             # Number of data points generated for regression

noise_franke = 0.1              # Noise level in generated regression data

test_proportion = 0.2           # Proportion of data used for testing

#Generate Franke function data:
data_points, y = tools.GenerateDataFranke(ndata=ndata_franke, noise_str=noise_franke)

#Split Franke function data into test- and training data:
x_train, x_test, y_train, y_test = train_test_split(data_points, y, test_size=test_proportion, random_state=42)

#Format data into appropriate dimensions:
x_train = sky.format_nxp(x_train)
x_test = sky.format_nxp(x_test)
y_train = sky.format_nx1(y_train)
y_test = sky.format_nx1(y_test)
test_data = list(zip(x_test, y_test))




###############################################################################
####################### Compare NN to other models  ###########################
###############################################################################

if 0:
    # Input parameters:
    
    n_epochs = 100                  # (Maximum) number of epochs to train the network over
    
    batch_size = 30                  # Batch size of SGD algorithm
    
    learn_rate = 0.001                # Learning rate of SGD algorithm
    
    lamb = 0.00                        # Regularization parameter
    
    hidden_nodes = 100              # Number of nodes in hidden layer
    
    stop_threshhold = 10            # Number of epochs to average over for stopping condition
    
    
    np.random.seed(42)
    
    # Create a neural network. Input layer has two nodes (for x- and y- value).
    # Output layer has 1 node (estimated function value):
    net = sky.Network([2, hidden_nodes, 1])
    
    #Set the activation function of the output layer:
    act_func = sky.Linear()     # Defines an activation_function object.
    net.afs[-1] = act_func      # Sets last element in vector of act.functions to object.
    
    #Set biases:
    net.biases = [np.zeros(shape=(y,1)) for y in net.layer_sizes[1:]] 
    
    #Train the network:
    net.train(x_train, y_train, n_epochs=n_epochs, batch_size=batch_size, 
              learn_rate=learn_rate, test_data=test_data, lamb=lamb, 
              stop_threshhold=stop_threshhold)
    
    
    
    # Plot R2 for test and training data:
    #------------------------------------
    
    qp.plot_title = "$R^2$-score of Neural Network"
    
    qp.x_label = "Epoch"
    
    qp.y_label = "$R^2$ score"
    
    qp.reset()
    qp.add_plot(np.arange(len(net.R2_))+1, net.R2_, 'r', label = "Training Data")
    qp.add_plot(np.arange(len(net.R2_test))+1, net.R2_test, 'b', label = "Testing Data")
    qp.create_plot(1)
    
    
    
    # Plot MSE for test and training data:
    #------------------------------------
    
    qp.plot_title = "$MSE$ of Neural Network"
    
    qp.x_label = "Epoch"
    
    qp.y_label = "$MSE$"
    
    
    qp.reset()
    qp.add_plot(np.arange(len(net.mse_))+1, net.mse_, 'r', label = "Training Data")
    qp.add_plot(np.arange(len(net.mse_test))+1, net.mse_test, 'b', label = "Testing Data")
    qp.create_plot(2)



###############################################################################
############### Try different L2 regularization parameters ####################
###############################################################################

if 0:
    # Input parameters:
    
    qp.plot_title = "Training with L2 regularization"
    
    qp.x_label = "Epoch"
    
    qp.y_label = "$MSE$"
        
    
    lamb_vec = [0, 0.001, 0.01, 0.1, 1, 10]         # Regularization parameters to try out
    color_vec = ['k','b', 'g', 'y', 'r', 'm']
    
    
    n_epochs = 200                      # (Maximum) number of epochs to train the network over
    
    batch_size = 30                     # Batch size of SGD algorithm
    
    learn_rate = 0.1                   # Learning rate of SGD algorithm
    
    
    hidden_nodes = 100                  # Number of nodes in hidden layer
    
    stop_threshhold = 10                # Number of epochs to average over for stopping condition
    
    
    
    
    
    qp.reset()
    
    for (lamb, color) in zip(lamb_vec,color_vec):
        
        np.random.seed(42)
        
        # Create a neural network. Input layer has two nodes (for x- and y- value).
        # Output layer has 1 node (estimated function value):
        net = sky.Network([2, hidden_nodes, 1])
        
        #Set the activation function of the output layer:
        act_func = sky.Linear()     # Defines an activation_function object.
        net.afs[-1] = act_func      # Sets last element in vector of act.functions to object.
        
        #Set biases:
        net.biases = [np.zeros(shape=(y,1)) for y in net.layer_sizes[1:]] 
        
        #Train the network:
        net.train(x_train, y_train, n_epochs=n_epochs, batch_size=batch_size, 
                  learn_rate=learn_rate, test_data=test_data, lamb=lamb, 
                  stop_threshhold=stop_threshhold, prin=False)
        
        qp.add_plot(np.arange(len(net.mse_test))+1, net.mse_test, color, label = "$\lambda =$ " + str(lamb))
        
        message = "Finished for lambda = " + str(lamb) 
        print(message)
            
        print(net.R2_[-1])
        print(net.R2_test[-1])
    
    qp.create_plot(3)



###############################################################################
############### Try different L1 regularization parameters ####################
###############################################################################

if 0:
    # Input parameters:
    
    figure_num = 4    
    
    qp.plot_title = "Training with L1 regularization"
    
    qp.x_label = "Epoch"
    
    qp.y_label = "$MSE$"
    
    lamb_vec = [0, 0.001, 0.01, 0.1, 1, 10]         # Regularization parameters to try out
    color_vec = ['k', 'b', 'g', 'y', 'r', 'm']
    
    
    n_epochs = 200                      # (Maximum) number of epochs to train the network over
    
    batch_size = 30                     # Batch size of SGD algorithm
    
    learn_rate = 0.1                   # Learning rate of SGD algorithm
    
    
    hidden_nodes = 100                  # Number of nodes in hidden layer
    
    stop_threshhold = 10                # Number of epochs to average over for stopping condition
    
    
    
    
    
    qp.reset()
    
    for (lamb, color) in zip(lamb_vec,color_vec):
        
        np.random.seed(42)
        
        # Create a neural network. Input layer has two nodes (for x- and y- value).
        # Output layer has 1 node (estimated function value):
        net = sky.Network([2, hidden_nodes, 1])
        
        #Set regularization to L1:
        net.regularization_type = sky.L1
        
        #Set the activation function of the output layer:
        act_func = sky.Linear()     # Defines an activation_function object.
        net.afs[-1] = act_func      # Sets last element in vector of act.functions to object.
        
        #Set biases:
        net.biases = [np.zeros(shape=(y,1)) for y in net.layer_sizes[1:]] 
        
        #Train the network:
        net.train(x_train, y_train, n_epochs=n_epochs, batch_size=batch_size, 
                  learn_rate=learn_rate, test_data=test_data, lamb=lamb, 
                  stop_threshhold=stop_threshhold, prin=False)
        
        qp.add_plot(np.arange(len(net.mse_test))+1, net.mse_test, color, label = "$\lambda =$ " + str(lamb))
        
        message = "Finished for lambda = " + str(lamb) 
        print(message)
        
        print(net.R2_[-1])
        print(net.R2_test[-1])
            
    
    
    qp.create_plot(figure_num)





###############################################################################
######################### Trying out Leaky RELU  ##############################
###############################################################################

if 1:
    # Input parameters:
    
    figure_num = 5    
    
    qp.plot_title = "Training with Leaky RELU"
    
    qp.x_label = "Epoch"
    
    qp.y_label = "$MSE$"
    
    leak_params = [0, 0.001, 0.01, 0.1, 1, 10]                    # Regularization parameters to try out
    color_vec = ['k' ,'b', 'g', 'y', 'r', 'm']
    
    lamb = 0
    
    n_epochs = 100                      # (Maximum) number of epochs to train the network over
    
    batch_size = 30                     # Batch size of SGD algorithm
    
    learn_rate = 0.01                   # Learning rate of SGD algorithm
    
    
    hidden_nodes = 100                  # Number of nodes in hidden layer
    
    stop_threshhold = 10                # Number of epochs to average over for stopping condition
    
    
    
    
    qp.reset()
    
    for (param, color) in zip(leak_params,color_vec):
        
        np.random.seed(42)
        
        # Create a neural network. Input layer has two nodes (for x- and y- value).
        # Output layer has 1 node (estimated function value):
        net = sky.Network([2, hidden_nodes, 1])
        
        #Set activation function for hidden layer_
        act_func = sky.Leaky_RELU(param)
        net.afs[-2] = act_func
        
        #Set the activation function of the output layer:
        act_func = sky.Linear()     # Defines an activation_function object.
        net.afs[-1] = act_func      # Sets last element in vector of act.functions to object.
        
        #Set biases:
        net.biases = [np.zeros(shape=(y,1)) for y in net.layer_sizes[1:]] 
        
        #Train the network:
        net.train(x_train, y_train, n_epochs=n_epochs, batch_size=batch_size, 
                  learn_rate=learn_rate, test_data=test_data, lamb=lamb, 
                  stop_threshhold=stop_threshhold, prin=True)
        
        qp.add_plot(np.arange(len(net.mse_test))+1, net.mse_test, color, label = "$a =$ " + str(param))
        
        message = "Finished for leak_param = " + str(param) 
        print(message)
            
    
    
    qp.create_plot(figure_num)



###############################################################################
####### Try different L2 regularization parameters for leaky RELU #############
###############################################################################

if 1:
    # Input parameters:
    
    qp.plot_title = "Training with L2 regularization and RELU activation"
    
    qp.x_label = "Epoch"
    
    qp.y_label = "$MSE$"
        
    
    lamb_vec = [0, 0.001, 0.01, 0.1, 1, 10]         # Regularization parameters to try out
    color_vec = ['k','b', 'g', 'y', 'r', 'm']
    
    leak_param = 0.00
    
    n_epochs = 100                      # (Maximum) number of epochs to train the network over
    
    batch_size = 30                     # Batch size of SGD algorithm
    
    learn_rate = 0.01                   # Learning rate of SGD algorithm
    
    
    hidden_nodes = 100                  # Number of nodes in hidden layer
    
    stop_threshhold = 10                # Number of epochs to average over for stopping condition
    
    
    
    
    
    qp.reset()
    
    for (lamb, color) in zip(lamb_vec,color_vec):
        
        np.random.seed(42)
        
        # Create a neural network. Input layer has two nodes (for x- and y- value).
        # Output layer has 1 node (estimated function value):
        net = sky.Network([2, hidden_nodes, 1])
        
        #Set activation function for hidden layer_
        act_func = sky.Leaky_RELU(leak_param)
        net.afs[-2] = act_func
        
        #Set the activation function of the output layer:
        act_func = sky.Linear()     # Defines an activation_function object.
        net.afs[-1] = act_func      # Sets last element in vector of act.functions to object.
        
        #Set biases:
        net.biases = [np.zeros(shape=(y,1)) for y in net.layer_sizes[1:]] 
        
        #Train the network:
        net.train(x_train, y_train, n_epochs=n_epochs, batch_size=batch_size, 
                  learn_rate=learn_rate, test_data=test_data, lamb=lamb, 
                  stop_threshhold=stop_threshhold, prin=True)
        
        qp.add_plot(np.arange(len(net.mse_test))+1, net.mse_test, color, label = "$\lambda =$ " + str(lamb))
        
        message = "Finished for lambda = " + str(lamb) 
        print(message)
            
        # print(net.R2_[-1])
        # print(net.R2_test[-1])
    
    qp.create_plot(6)











