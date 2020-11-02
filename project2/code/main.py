#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pprint
plt.rc('font',family='serif')

def get_data(filename):
    var = filename.split("_")
    var = var[1:5]                      # ep, bs, lr, gm
    cost = np.loadtxt(filename+".csv")
    info = np.loadtxt(filename+".txt")
    stat = info[0:2]                    # R2, mse
    return cost, var, stat

def plot_epoch(cost, var, stat):
    n_batches = cost.shape[0]
    n_epochs = cost.shape[1]
    epochs = np.arange(1, n_epochs+1)
    ll =0
    plt.plot(epochs, cost[0], label=r"$\alpha$: "+var[2]+" (R2={: .2f})".format(stat[0]))


def plot1(PATH, filenames):
    cost, var, stat = 0, 0, 0
    fig = plt.figure()
    plt.grid()
    for name in filenames:
        cost, var, stat = get_data(PATH+"data/"+name)
        plot_epoch(cost, var, stat)

    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.legend()
    fig.savefig(PATH+"figures/"+"SGD_LR_bs_{:}.png".format(var[1]))
    plt.show()

if __name__ == '__main__':
    PATH = "output/"

    filenames = ["SGDLOG_100_5_0.1_standard_",
                 "SGDLOG_100_5_0.01_standard_",
                 "SGDLOG_100_5_0.001_standard_",
                 "SGDLOG_100_5_0.0001_standard_",
                 "SGDLOG_100_5_1e-05_standard_"]
    plot1(PATH, filenames)
