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
        
    
    lamb_vec = [1, 10]         # Regularization parameters to try out
    color_vec = ['r', 'm']     # ['k','b', 'g', 'y', 
    
    
    n_epochs = 100                      # (Maximum) number of epochs to train the network over
    
    batch_size = 30                     # Batch size of SGD algorithm
    
    learn_rate = 1                   # Learning rate of SGD algorithm
    
    
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
                  accuracy=True, prin=True)
        
        
        
        qp.add_plot(np.arange(len(net.accuracies))+1, net.accuracies, color, label = "$\lambda =$ " + str(lamb))
        
        message = "Finished for lambda = " + str(lamb) 
        print(message)
            

    
    qp.create_plot(fig_num)   



###############################################################################
############### Try different L1 regularization parameters ####################
###############################################################################

if 0:
    # Input parameters:
    
    fig_num = 3
        
    qp.plot_title = "Training with L1 regularization"
    
    qp.x_label = "Epoch"
    
    qp.y_label = "Accuracy"
        
    
    lamb_vec = [1, 10]         # Regularization parameters to try out
    color_vec = ['r', 'm'] #['k','b', 'g', 'y', 
    
    
    n_epochs = 100                      # (Maximum) number of epochs to train the network over
    
    batch_size = 30                     # Batch size of SGD algorithm
    
    learn_rate = 1                   # Learning rate of SGD algorithm
    
    
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
                  accuracy=True, prin=True)
        
        
        
        qp.add_plot(np.arange(len(net.accuracies))+1, net.accuracies, color, label = "$\lambda =$ " + str(lamb))
        
        message = "Finished for lambda = " + str(lamb) 
        print(message)
            

    
    qp.create_plot(fig_num)  




###############################################################################
######################### Trying out Leaky RELU  ##############################
###############################################################################

if 0:
    # Input parameters:
    
    figure_num = 4    
    
    qp.plot_title = "Training with Leaky RELU"
    
    qp.x_label = "Epoch"
    
    qp.y_label = "$Accuracy$"
    
    leak_params = [0, 0.001, 0.01, 0.1, 1, 10]                    # Regularization parameters to try out
    color_vec = ['k' ,'b', 'g', 'y', 'r', 'm']
    
    lamb = 1
    
    n_epochs = 100                      # (Maximum) number of epochs to train the network over
    
    batch_size = 30                     # Batch size of SGD algorithm
    
    learn_rate = 3                   # Learning rate of SGD algorithm
    
    
    hidden_nodes = 30                  # Number of nodes in hidden layer
    
    stop_threshhold = 10                # Number of epochs to average over for stopping condition
    
    
    
    
    qp.reset()
    
    for (param, color) in zip(leak_params,color_vec):
        
        np.random.seed(42)
        
        # Create a neural network. Input layer has two nodes (for x- and y- value).
        # Output layer has 1 node (estimated function value):
        net = sky.Network([784, hidden_nodes, 10])
        
        #Set activation function for hidden layer_
        act_func = sky.Leaky_RELU(param)
        net.afs[-2] = act_func
        
        #Set the activation function of the output layer:
        act_func = sky.Sigmoid()     # Defines an activation_function object.
        net.afs[-1] = act_func      # Sets last element in vector of act.functions to object.
        
        #Set biases:
        net.biases = [np.zeros(shape=(y,1)) for y in net.layer_sizes[1:]] 
        
        #Train the network:
        net.train(x_train, y_train, n_epochs=n_epochs, batch_size=batch_size, 
                  learn_rate=learn_rate, test_data=test_data, lamb=lamb, 
                  stop_threshhold=stop_threshhold, stop_criterion="accuracy", 
                  accuracy=True, prin=True)
        
        qp.add_plot(np.arange(len(net.accuracies))+1, net.accuracies, color, label = "$a =$ " + str(param))
        
        message = "Finished for leak_param = " + str(param) 
        print(message)
            
    
    
    qp.create_plot(figure_num)



###############################################################################
###########           Testing Logistic regression            ##################
###############################################################################

if 0:

    # Input parameters:
    
    n_epochs = 50                  # (Maximum) number of epochs to train the network over
    
    batch_size = 30                  # Batch size of SGD algorithm
    
    learn_rates = [0.003, 0.03, 0.3]                # Learning rate of SGD algorithm
    colors = ['b', 'g', 'r']
    
    
    lamb = 1.00                        # Regularization parameter
    
    hidden_nodes = 30              # Number of nodes in hidden layer
    
    stop_threshhold = 10            # Number of epochs to average over for stopping condition

    qp.reset()
    
    for learn_rate, color in zip(learn_rates, colors):
        np.random.seed(42)
        
        # Create a neural network. Input layer has two nodes (for x- and y- value).
        # Output layer has 1 node (estimated function value):
        net = sky.Network([784, 10])
        
        # Set the cost function to cross entropy
        net.cost = sky.CrossEntropy()
            
        #Set the activation function of the output layer:
        act_func = sky.Softmax()     # Defines an activation_function object.
        net.afs[-1] = act_func      # Sets last element in vector of act.functions to object.
    
    
        #Train the network:
        net.train(x_train, y_train, n_epochs=n_epochs, batch_size=batch_size, 
                  learn_rate=learn_rate, test_data=test_data, lamb=lamb, 
                  stop_threshhold=stop_threshhold, stop_criterion="accuracy", accuracy=True)
        
        
        
        # Plot R2 for test and training data:
        #------------------------------------
        
        qp.plot_title = "Training of Logistic Regression Algorithm"
        
        qp.x_label = "Epoch"
        
        qp.y_label = "Accuracy"
        
        
        qp.add_plot(np.arange(len(net.accuracies))+1, net.accuracies, color, label = "Learn Rate = " + str(learn_rate))
        
    qp.create_plot(5)



###############################################################################
###############          Comparing with Scikit learn       ####################
###############################################################################

def format_scikit(datapoints):
    
    new_datapoints = list(np.zeros(len(datapoints)))
    
    for i, datapoint in enumerate(datapoints):
        
        new_datapoints[i] = datapoint.T[0]
    
    return new_datapoints


def format_test(datapoints):
    
    new_datapoints = list(np.zeros(len(datapoints)))
    
    for i, datapoint in enumerate(datapoints):
        
        new_datapoints[i] = np.zeros(10)
        new_datapoints[i][ datapoints[i]  ] = 1
    
    return new_datapoints

        


if 1:

    from sklearn.neural_network import MLPClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(n_samples=100, random_state=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, stratify=y, random_state=1)
    
    # X_train =  np.array([ [4,6,7,8], [2,5,8,9], [5,78,96,4], [6,7,3,2]      ])
    # Y_train =  np.array([ [1,0,0,0], [0,0,1,0], [0,1,0,0], [0,1,0,0]      ])
    
    x_scikit = format_scikit(x_train)
    y_scikit = format_scikit(y_train)
    
    x_test, y_test = sky.split_data(test_data)
    
    y_test_formatted = format_test(y_test)
    
    y_scikit_test = y_test_formatted    #format_scikit(y_test_formated)
    x_scikit_test = format_scikit(x_test)
    
    
    
     
    clf = MLPClassifier(random_state=1, max_iter=300).fit(x_scikit, y_scikit)
     
    
    clf.score(x_scikit_test, y_scikit_test)
    
    
    


































