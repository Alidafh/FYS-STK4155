#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 22:54:43 2020

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
################### Load and prepare Classification Data ######################
###############################################################################

# Load mnist data and save as training, testing and validation batches:
training_data, validation_data, test_data =  mnist_loader.load_data_wrapper()   

# Format data:
training_data = list(training_data)
validation_data = list(validation_data)
test_data = list(test_data)
x_train,y_train = sky.split_data(training_data)             # x is collection of independent variables, y the collection of expected outputs.






###############################################################################
###################           Testing the NN             ######################
###############################################################################

if 0:

    # Input parameters:
    
    n_epochs = 100                  # (Maximum) number of epochs to train the network over
    
    batch_size = 30                  # Batch size of SGD algorithm
    
    learn_rate = 3                # Learning rate of SGD algorithm
    
    lamb = 0.00                        # Regularization parameter
    
    hidden_nodes = 30              # Number of nodes in hidden layer
    
    stop_threshhold = 10            # Number of epochs to average over for stopping condition


    np.random.seed(42)
    
    # Create a neural network. Input layer has two nodes (for x- and y- value).
    # Output layer has 1 node (estimated function value):
    net = sky.Network([784, hidden_nodes, 10])
    
    #Set the activation function of the output layer:
    act_func = sky.Softmax()     # Defines an activation_function object.
    net.afs[-1] = act_func      # Sets last element in vector of act.functions to object.


    #Train the network:
    net.train(x_train, y_train, n_epochs=n_epochs, batch_size=batch_size, 
              learn_rate=learn_rate, test_data=test_data, lamb=lamb, 
              stop_threshhold=stop_threshhold, stop_criterion="accuracy", accuracy=True)
    
    
    
    # Plot R2 for test and training data:
    #------------------------------------
    
    qp.plot_title = "Training of Neural Network"
    
    qp.x_label = "Epoch"
    
    qp.y_label = "Accuracy"
    
    qp.reset()
    qp.add_plot(np.arange(len(net.accuracies_train))+1, net.accuracies_train, 'r', label = "Training Data")
    qp.add_plot(np.arange(len(net.accuracies))+1, net.accuracies, 'b', label = "Testing Data")
    qp.create_plot(1)




###############################################################################
############### Try different L2 regularization parameters ####################
###############################################################################

if 0:
    # Input parameters:
    
    fig_num = 2
        
    qp.plot_title = "Training with L2 regularization"
    
    qp.x_label = "Epoch"
    
    qp.y_label = "Accuracy"
        
    
    lamb_vec = [0, 0.001, 0.01, 0.1, 1, 10]         # Regularization parameters to try out
    color_vec = ['k','b', 'g', 'y', 'r', 'm']
    
    
    n_epochs = 4                      # (Maximum) number of epochs to train the network over
    
    batch_size = 30                     # Batch size of SGD algorithm
    
    learn_rate = 3                   # Learning rate of SGD algorithm
    
    
    hidden_nodes = 30                  # Number of nodes in hidden layer
    
    stop_threshhold = 10                # Number of epochs to average over for stopping condition
    
    
    
    
    
    qp.reset()
    
    for (lamb, color) in zip(lamb_vec,color_vec):
        
        np.random.seed(42)
        
        # Create a neural network. Input layer has two nodes (for x- and y- value).
        # Output layer has 1 node (estimated function value):
        net = sky.Network([784, hidden_nodes, 10])
        
        
        #Set the activation function of the output layer:
        act_func = sky.Softmax()     # Defines an activation_function object.
        net.afs[-1] = act_func      # Sets last element in vector of act.functions to object.
    
    
        #Train the network:
        net.train(x_train, y_train, n_epochs=n_epochs, batch_size=batch_size, 
                  learn_rate=learn_rate, test_data=test_data, lamb=lamb, 
                  stop_threshhold=stop_threshhold, stop_criterion="accuracy", 
                  accuracy=True, prin=False)
        
        
        
        qp.add_plot(np.arange(len(net.accuracies))+1, net.accuracies, color, label = "$\lambda =$ " + str(lamb))
        
        message = "Finished for lambda = " + str(lamb) 
        print(message)
            

    
    qp.create_plot(fig_num)   



###############################################################################
############### Try different L1 regularization parameters ####################
###############################################################################

if 1:
    # Input parameters:
    
    fig_num = 3
        
    qp.plot_title = "Training with L1 regularization"
    
    qp.x_label = "Epoch"
    
    qp.y_label = "Accuracy"
        
    
    lamb_vec = [0, 0.001, 0.01, 0.1, 1, 10]         # Regularization parameters to try out
    color_vec = ['k','b', 'g', 'y', 'r', 'm']
    
    
    n_epochs = 2                      # (Maximum) number of epochs to train the network over
    
    batch_size = 30                     # Batch size of SGD algorithm
    
    learn_rate = 3                   # Learning rate of SGD algorithm
    
    
    hidden_nodes = 30                  # Number of nodes in hidden layer
    
    stop_threshhold = 10                # Number of epochs to average over for stopping condition
    
    
    
    
    
    qp.reset()
    
    for (lamb, color) in zip(lamb_vec,color_vec):
        
        np.random.seed(42)
        
        # Create a neural network. Input layer has two nodes (for x- and y- value).
        # Output layer has 1 node (estimated function value):
        net = sky.Network([784, hidden_nodes, 10])
        
        #Set regularization to L1:
        net.regularization_type = sky.L1
        
        #Set the activation function of the output layer:
        act_func = sky.Softmax()     # Defines an activation_function object.
        net.afs[-1] = act_func      # Sets last element in vector of act.functions to object.
    
    
        #Train the network:
        net.train(x_train, y_train, n_epochs=n_epochs, batch_size=batch_size, 
                  learn_rate=learn_rate, test_data=test_data, lamb=lamb, 
                  stop_threshhold=stop_threshhold, stop_criterion="accuracy", 
                  accuracy=True, prin=False)
        
        
        
        qp.add_plot(np.arange(len(net.accuracies))+1, net.accuracies, color, label = "$\lambda =$ " + str(lamb))
        
        message = "Finished for lambda = " + str(lamb) 
        print(message)
            

    
    qp.create_plot(fig_num)  


















