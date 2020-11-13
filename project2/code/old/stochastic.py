#!/usr/bin/python
import numpy as np
import tools as tools
from sklearn.model_selection import train_test_split
from regression import OLS, Ridge
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
plt.rc('font',family='serif')

def get_data(filename):
    var = filename.split("_")
    var = var[1:6]                      # d, ep, bs, lr, gm

    cost = np.loadtxt(filename+".csv")
    info = np.loadtxt(filename+".txt")

    info_sgd = info[0,:]
    info_basic = info[1,:]

    stat_sgd = info[0,:2]
    beta_sgd = info[0, 2:]

    stat_basic = info[1,:2]
    beta_basic = info[1,2:]

    return cost, var, stat_sgd, beta_sgd, stat_basic, beta_basic

def get_data(filename):
    var = filename.split("_")
    var = var[1:6]                      # d, ep, bs, lr, gm

    cost = np.loadtxt(filename+".csv")
    info = np.loadtxt(filename+".txt")

    info_sgd = info[0,:]
    info_basic = info[1,:]

    stat_sgd = info[0,:2]
    beta_sgd = info[0, 2:]

    stat_basic = info[1,:2]
    beta_basic = info[1,2:]

    return cost, var, stat_sgd, beta_sgd, stat_basic, beta_basic

def plot_lr_epoch(PATH, filenames):
    cost, var, stat = 0, 0, 0
    fig = plt.figure()
    plt.grid()
    for name in filenames:
        cost, var, stat, beta, stat_basic, beta_basic = get_data(PATH+"data/"+name)
        n_epochs = cost.shape[1]
        epochs = np.arange(1, n_epochs+1)
        plt.plot(epochs, cost[0], label=r"$\alpha$: "+var[3]+" (R2={: .2f})".format(stat[0]))

    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.legend()
    fig.savefig(PATH+"figures/"+"SGD_LR_bs_{:}_d_{:}.png".format(var[2], var[0]))
    #plt.close()
    plt.show()

def plot_mse(PATH, filenames):
    len_d = len(filenames)
    degrees = np.zeros(len_d)
    mse_sgd = np.zeros(len_d)
    r2_sgd = np.zeros(len_d)
    mse_basic = np.zeros(len_d)
    r2_basic = np.zeros(len_d)

    for name in filenames:
        cost, var, stat_sgd, beta_sgd, stat_basic, beta_basic = get_data(PATH+"data/"+name)
        d = int(var[0])
        degrees[d-1] = d
        r2_sgd[d-1] = float(stat_sgd[0])
        mse_sgd[d-1] = float(stat_sgd[1])

        r2_basic[d-1] = float(stat_basic[0])
        mse_basic[d-1] = float(stat_basic[1])

    fig, ax = plt.subplots(nrows=2,
                           ncols=1,
                           sharex="col",
                           sharey=False,
                           figsize=[6.4, 6.4],
                           constrained_layout=True) #[6.4, 4.8].

    ax[0].set_ylabel("MSE",fontsize = 10, fontname = "serif" )
    ax[0].plot(degrees, mse_sgd, label = "Stochastic Gradient Descent")
    ax[0].plot(degrees, mse_basic, label = "Linear regresson")

    ax[1].set_ylabel("R2",fontsize = 10, fontname = "serif" )
    ax[1].plot(degrees, r2_sgd, label = "Stochastic Gradient Descent")
    ax[1].plot(degrees, r2_basic, label = "Linear regresson")

    lines = []
    labels = []

    for ax in fig.axes:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)

    fig.legend(lines[:2], labels[:2], loc = 'upper right')
    plt.xlabel("Model complexity (Degrees)")
    fig.savefig(PATH+"figures/"+"SGD_MSE_complexity_lr_{:}.png".format(var[3]))
    plt.show()


def main_LR():
    PATH = "output/"

    filenames = ["SGDLOG_4_100_5_0.1_standard_",
                 "SGDLOG_4_100_5_0.01_standard_",
                 "SGDLOG_4_100_5_0.001_standard_",
                 "SGDLOG_4_100_5_0.0001_standard_",
                 "SGDLOG_4_100_5_1e-05_standard_",
                 "SGDLOG_4_100_5_schedule_standard_"]

    #filenames = ["SGDLOG_4_100_5_0.1_standard_"]
    plot_lr_epoch(PATH, filenames)


def main_complexity(lr):
    PATH = "output/"
    filenames = ["SGDLOG_1_100_5_{:}_standard_".format(lr),
                 "SGDLOG_2_100_5_{:}_standard_".format(lr),
                 "SGDLOG_3_100_5_{:}_standard_".format(lr),
                 "SGDLOG_4_100_5_{:}_standard_".format(lr),
                 "SGDLOG_5_100_5_{:}_standard_".format(lr),
                 "SGDLOG_6_100_5_{:}_standard_".format(lr),
                 "SGDLOG_7_100_5_{:}_standard_".format(lr),
                 "SGDLOG_8_100_5_{:}_standard_".format(lr),
                 "SGDLOG_9_100_5_{:}_standard_".format(lr),
                 "SGDLOG_10_100_5_{:}_standard_".format(lr)]

    plot_mse(PATH, filenames)

if __name__ == '__main__':
    #main_LR()
    main_complexity(lr = 0.01)
    #main_complexity(lr = 0.001)
    #cost, var, stat, beta = get_data("output/data/SGDLOG_1_100_5_0.01_standard_")
