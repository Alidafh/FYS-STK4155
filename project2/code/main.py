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

def read_file1(filename):
    data = np.loadtxt(filename)
    n_batches = data.shape[0]
    n_epochs = data.shape[1]
    cols = ["epoch_{:}".format(i) for i in range(1, n_epochs+1)]
    ind = np.arange(1, n_batches+1)
    df = pd.DataFrame(data, columns=cols, index = ind)
    print(df)
    e1 = df.get(["epoch_1"])
    print(e1)


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
    loss_ep = loss[n_batches-1]
    plt.plot(epochs, loss_ep, label="learn_rate: "+lr+" (R2={: .2f})".format(r2))

def main():
    fig = plt.figure()
    plt.grid()
    plot_epoch("output/data/SGDLOG_100_5_0.1_standard_")
    plot_epoch("output/data/SGDLOG_100_5_0.01_standard_")
    plot_epoch("output/data/SGDLOG_100_5_0.001_standard_")
    plot_epoch("output/data/SGDLOG_100_5_0.0001_standard_")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    fig.savefig("output/figures/SGD_LR_bs5.png")
    plt.show()

if __name__ == '__main__':
    main()
    #plot_epoch("output/data/SGDLOG_3_100_0.1_standard_.txt")
    #read_file("output/data/SGDLOG_3_100_0.1_standard_.txt")
