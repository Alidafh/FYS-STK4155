#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pprint
plt.rc('font',family='serif')

def read_file(filename):
    loss = np.loadtxt(filename)
    n_batches = loss.shape[0]
    n_epochs = loss.shape[1]
    return loss

def plot_epoch(filename):
    var = filename.split("_")
    ep, bs, lr, gm = var[1], var[2], var[3], var[4]

    loss = np.loadtxt(filename+".csv")
    info = np.loadtxt(filename+".txt")
    r2 = info[0]
    mse = info[1]
    n_batches = loss.shape[0]
    n_epochs = loss.shape[1]

    epochs = np.arange(1, n_epochs+1)
    #loss_ep = loss[n_batches-1]
    loss_ep = loss[0]
    plt.plot(epochs, loss_ep, label="learn_rate: "+lr+" (R2={: .2f})".format(r2))


def main(PATH, filenames):
    var = filenames[0].split("_")
    ep, bs, lr, gm = var[1], var[2], var[3], var[4]

    fig = plt.figure()
    plt.grid()

    for name in filenames:
        plot_epoch(PATH+"data/"+name)

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    #print(PATH+"figures/"+"SGD_LR_bs{:}.png".format(bs))
    fig.savefig(PATH+"figures/"+"SGD_LR_bs_{:}.png".format(bs))
    plt.show()

if __name__ == '__main__':
    PATH = "output/"
    #filenames = ["SGDLOG_100_5_0.1_standard_",
    #             "SGDLOG_100_5_0.01_standard_",
    #             "SGDLOG_100_5_0.001_standard_"]

    filenames = ["SGDLOG_100_5_0.01_standard_",
                 "SGDLOG_100_5_0.0001_standard_",
                 "SGDLOG_100_5_1e-05_standard_"]
    main(PATH, filenames)
