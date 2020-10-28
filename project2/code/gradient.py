#!/usr/bin/python
import argparse

import numpy as np
import tools as tools
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from regression import OLS, Ridge
import sys

def stochastic(p=False):
    #Generate the data for the Franke function and divide into training and test
    #input, y = tools.GenerateDataFranke(ndata=1000, noise_str=0.1)
    #X = PolynomialFeatures(degree=d).fit_transform(input)
    X, y = tools.GenerateDataLine(100)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_test = tools.scale_X(X_train, X_test)

    # Set up the linear model
    if r =="OLS": model = OLS()
    if r == "Ridge": model=Ridge(lamb)

    loss = model.SGD(X_train, y_train, n_epochs=ep, batch_size=bs, learn_rate=lr, gamma=gm, prin=p)
    print("Beta : ", model.beta, sep='\n')
    print("r2 : ", model.r2score(X_test, y_test), '\n')

    info = np.array([model.r2score(X_test, y_test), model.mse(X_test, y_test)])

    # saves the losses to file
    lr_s = lr if lr else "schedule"
    gm_s = gm if gm else "standard"
    filename = "output/data/SGDLOG_{:}_{:}_{:}_{:}_.csv".format(ep, bs, lr_s, gm_s)
    filename2 = "output/data/SGDLOG_{:}_{:}_{:}_{:}_.txt".format(ep, bs, lr_s, gm_s)
    np.savetxt(filename, loss, fmt="%.8f")
    np.savetxt(filename2, info, fmt="%.8f")

##############################################################################

def main():
    # Set up parser
    description =  """Use Stochastic Gradient Descent to find the beta values
                    either for OLS or Ridge loss functions. A log file of the
                    loss for each epoch number and batch number is created and
                    stored in output/data/SGDLOG_*.txt"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-r', type=str, metavar='--method', action='store', default="OLS", help='The regression method, options are [OLS] and [Ridge]')
    parser.add_argument('-l', type=float, metavar='--lambda', action='store', default=0.001, help='The lambda value for ridge regression')
    parser.add_argument('-d', type=float, metavar='--gamma', action='store', default=4, help='Polynomial degree of design matrix')
    parser.add_argument('-ep', type=int, metavar='--n_epochs', action='store', default=100, help='The number of epochs')
    parser.add_argument('-bs', type=int, metavar='--batch_size', action='store', default=5, help='Size of the minibatches')
    parser.add_argument('-lr', type=float, metavar='--learn_rate', action='store', default=None, help='The learning rate')
    parser.add_argument('-gm', type=float, metavar='--gamma', action='store', default=None, help='The gamma value for momentum')

    args = parser.parse_args()

    r, lamb, d, ep, bs, lr, gm = args.r, args.l,args.d, args.ep, args.bs, args.lr, args.gm
    #print(r, lamb, d, ep, bs, lr, gm )
    print("Method:     ", r)
    print("Polydegree: ", d)
    print("Epochs:     ", ep)
    print("Batch size: ", bs)
    print("Learn rate: ", lr)
    print("Gamma:      ", gm)
    print()
    return r, lamb, d, ep, bs, lr, gm

if __name__ == "__main__":
    r, lamb, d, ep, bs, lr, gm = main()
    stochastic(p=False)
