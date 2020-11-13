#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 1 09:14:21 2020

Script to easily generate results using SGD with different parameter choises
specified in the command line. Path to where results should be stored must be
specified on line 119.

@author: Alida Hardersen
"""
import argparse
import numpy as np
import tools as tools
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import skynet as reg
import sys

def stochastic(PATH):
    print("Method:     ", r)
    print("Lambda:     ", lamb, "(Only if Ridge)")
    print("Polydegree: ", d)
    print("Epochs:     ", ep)
    print("Batch size: ", bs)
    print("Learn rate: ", lr)
    print("Gamma:      ", gm)
    print()

    #Generate the data for the Franke function and divide into training and test
    input, y = tools.GenerateDataFranke(ndata=1000, noise_str=0.1)
    X = tools.PolyDesignMatrix(input.T, d=d)[0]

    X_train, X_test, y_train, y_test = tools.split_scale(X, y)

    # Set up the linear model
    if r =="OLS": model = reg.OLS()
    if r == "Ridge": model = reg.Ridge(lamb)

    loss = model.SGD(X_train, y_train, n_epochs=ep, batch_size=bs, learn_rate=lr,
                    gamma=gm, prin=p, accuracy=False, test_data=None, lamb = 0,
                    stop_threshhold=None, stop_criterion=None)

    print("Stochastic Gradient Descent")
    print("-----------------------------------")
    print("Beta : ", model.beta, sep='\n')
    print("r2 : ", model.r2score(X_test, y_test))
    print("mse: ", model.mse(X_test, y_test), '\n')

    info = np.zeros((2, len(model.beta)+2))
    info[0, 0] = model.r2score(X_test, y_test)
    info[0, 1] = model.mse(X_test, y_test)
    info[0, 2:] = model.beta

    ## Do normal regression to compare
    if r =="OLS": model_basic = reg.OLS()
    if r == "Ridge": model_basic = reg.Ridge(lamb)

    model_basic.fit(X_train, y_train)
    print()
    print("Ordinary Least Squares:")
    print("-----------------------------------")
    print("Beta : ", model_basic.beta, sep='\n')
    print("r2 : ", model_basic.r2score(X_test, y_test))
    print("mse: ", model_basic.mse(X_test, y_test), '\n')
    print("#############################################################")
    print()
    info[1, 0] = model_basic.r2score(X_test, y_test)
    info[1, 1] = model_basic.mse(X_test, y_test)
    info[1, 2:] = model_basic.beta


    # save the losses to file
    lr_s = lr if lr else "schedule"
    gm_s = gm if gm else "standard"

    if r=="OLS":
        filename = PATH+"SGDLOG_{:}_{:}_{:}_{:}_{:}_{:}_.csv".format(r, d, ep, bs, lr_s, gm_s)
        filename2 = PATH+"SGDLOG_{:}_{:}_{:}_{:}_{:}_{:}_.txt".format(r, d, ep, bs, lr_s, gm_s)
    if r=="Ridge":
        filename = PATH+"SGDLOG_{:}_{:}_{:}_{:}_{:}_{:}_{:}_.csv".format(r, lamb, d, ep, bs, lr_s, gm_s)
        filename2 = PATH+"SGDLOG_{:}_{:}_{:}_{:}_{:}_{:}_{:}_.txt".format(r, lamb, d, ep, bs, lr_s, gm_s)

    np.savetxt(filename, loss, fmt="%.8f")
    np.savetxt(filename2, info, fmt="%.8f")


##############################################################################

def main():
    # Set up parser
    description =  """Use Stochastic Gradient Descent to find the beta values
                    either for OLS or Ridge loss functions. A log file of the
                    loss for each epoch number and batch number is created and
                    stored in output/SGDLOG_*.txt"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-r', type=str, metavar='--method', action='store', default="OLS",
                    help='The regression method, options are [OLS] and [Ridge], default=OLS')
    parser.add_argument('-l', type=float, metavar='--lambda', action='store', default=0.001,
                    help='The lambda value for ridge regression, default=0.001')
    parser.add_argument('-d', type=int, metavar='--degree', action='store', default=4,
                    help='Polynomial degree of design matrix, default=4')
    parser.add_argument('-ep', type=int, metavar='--n_epochs', action='store', default=100,
                    help='The number of epochs, default=100')
    parser.add_argument('-bs', type=int, metavar='--batch_size', action='store', default=5,
                    help='Size of the minibatches, default=5')
    parser.add_argument('-lr', type=float, metavar='--learn_rate', action='store', default=None,
                    help='The learning rate, default=None')
    parser.add_argument('-gm', type=float, metavar='--gamma', action='store', default=None,
                    help='The gamma value for momentum, default=None')
    parser.add_argument('-p', type=str, metavar='--print', action='store', default="True",
                    help='If the loss should be printed, default=True')

    args = parser.parse_args()

    r, lamb, d, ep, bs, lr, gm, p = args.r, args.l,args.d, args.ep, args.bs, args.lr, args.gm, args.p
    pr = False if p =="False" else True
    return r, lamb, d, ep, bs, lr, gm, pr

if __name__ == "__main__":
    r, lamb, d, ep, bs, lr, gm, p = main()
    stochastic(PATH="../output/")
