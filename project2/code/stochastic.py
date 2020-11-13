#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 22:22:22 2020

Visualize the data generated using run_gradient.py

@author: Alida Hardersen
"""
import numpy as np
import tools as tools
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def get_data(filename):
    def make_dict(var, stat, beta):
        dict = {}
        keys = ["reg", "d", "ep", "bs", "lr", "gm"]
        if var[0] == "Ridge":
            keys = ["reg", "lamb", "d", "ep", "bs", "lr", "gm"]

        for key, v in zip(keys, var):
            dict[key] = v

        keys2 = ["r2", "mse"]
        for key, s in zip(keys2,stat):
            dict[key] = s

        dict["beta"] = beta
        return dict

    var = filename.split("_")
    var = var[1:7]                      # reg, (lamb), d, ep, bs, lr, gm

    cost = np.loadtxt(filename+".csv")
    info = np.loadtxt(filename+".txt")

    info_sgd = info[0,:]
    info_basic = info[1,:]

    stat_sgd = info[0,:2]   # r2, mse
    beta_sgd = info[0, 2:]  # beta

    stat_basic = info[1,:2]
    beta_basic = info[1,2:]

    dict_sgd = make_dict(var, stat_sgd, beta_sgd)
    dict_sgd["cost"] = cost
    dict_basic = make_dict(var, stat_basic, beta_basic)
    return dict_sgd, dict_basic

##############################################################################

def plot_lr_epoch(PATH, filenames, title=None):

    fig = plt.figure()
    #plt.grid()

    for name in filenames:
        sgd, basic = get_data(PATH+name)
        n_epochs = len(sgd["cost"])
        epochs = np.arange(1, n_epochs+1)
        plt.plot(epochs, sgd["cost"], label=r"$\alpha$: "+sgd["lr"]+" (R2:{: .2f}, MSE:{: .2f})".format(sgd["r2"], sgd["mse"]))

    #plt.title("SGD, polydegree: {:}, batch_size: {:}".format(sgd["d"], sgd["bs"]))
    plt.xlabel("Epochs")
    plt.ylabel("Training loss")
    plt.legend()
    if title: fig.savefig("../figures/"+title+".png")
    if not title: fig.savefig("../figures/"+"{:}_SGDLR_{:}_{:}_{:}_{:}_{:}.png".format(sgd["reg"],sgd["d"],sgd["ep"],sgd["bs"], sgd["gm"],len(filenames)))
    plt.show()


def plot_lr_epoch3(PATH, filenames1, filenames2, title=None):
    fig, ax = plt.subplots(nrows=2,
                           ncols=1,
                           sharex="col",
                           sharey=False,
                           figsize=[6.4, 6.4],
                           constrained_layout=True) #[6.4, 4.8].

    colors = ["tab:green", "tab:blue"]
    for name, c in zip(filenames1, colors):
        sgd, basic = get_data(PATH+name)
        n_epochs = len(sgd["cost"])
        epochs = np.arange(1, n_epochs+1)
        ax[0].plot(epochs, sgd["cost"], c, label=r"$\alpha$: "+sgd["lr"]+" (R2:{: .2f}, MSE:{: .2f})".format(sgd["r2"], sgd["mse"]))
        ax[0].set_ylabel("Training loss")
        ax[0].legend()

    for name, c in zip(filenames2, colors):
        sgd, basic = get_data(PATH+name)
        n_epochs = len(sgd["cost"])
        epochs = np.arange(1, n_epochs+1)
        ax[1].plot(epochs, sgd["cost"], c, label=r"$\alpha$: "+sgd["lr"]+r" $\gamma: $"+sgd["gm"]+" (R2:{: .2f}, MSE:{: .2f})".format(sgd["r2"], sgd["mse"]))
        ax[1].set_ylabel("Training loss")
        ax[1].legend()

    plt.xlabel("Epochs")
    if title: fig.savefig("../figures/"+title+".png")
    if not title: fig.savefig("../figures/"+"SGDLR_{:}_{:}_{:}_{:}_{:}.png".format(sgd["d"],sgd["ep"],sgd["bs"], sgd["gm"],len(filenames1)))
    plt.show()


def plot_lr_epoch_shedule(PATH, filenames, title=None):
    fig, ax = plt.subplots(nrows=2,
                           ncols=1,
                           sharex="col",
                           sharey=False,
                           figsize=[6.4, 6.4],
                           constrained_layout=True) #[6.4, 4.8].


    n = len(filenames)
    c=["tab:blue", "tab:green"]
    for i in range(n):
        name=filenames[i]
        sgd, basic = get_data(PATH+name)
        n_epochs = int(float(sgd["ep"]))
        epochs = np.arange(1, n_epochs+1)
        ax[0].set_ylabel("Training loss",fontsize = 10)
        ax[0].plot(epochs, sgd["cost"], color=c[i], label= r"$\alpha: $"+sgd["lr"]+"R2: {:.2f}, MSE: {:.2f}".format(sgd["r2"], sgd["mse"]))

    n_epoch = int(sgd["ep"])
    m = int(1024*0.8/5)

    lr = np.zeros((n_epoch, m))
    lr2 = np.zeros((n_epoch, m))

    for ep in range(n_epoch):
        for i in range(m):
            y1 = tools.learning_schedule(ep*m+i, 5, 50)
            y2 = tools.learning_schedule(ep+i, 5, 50)
            lr[ep, i] = y1
            lr2[ep,i] = y2

    x = np.linspace(0, n_epoch, len(lr.ravel()))
    ax[1].set_ylabel(r"Learn rate, $\alpha$", fontsize = 10)
    ax[1].plot(x, lr2.ravel(), color=c[0], label= r"$\alpha: $schedule2")
    ax[1].plot(x, lr.ravel(), color=c[1], label= r"$\alpha: $schedule")
    lines = []
    labels = []

    for ax in fig.axes:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)

    fig.legend(lines[:2], labels[:2], loc = 'upper right')
    plt.xlabel("Epochs")
    if title: fig.savefig("../figures/"+title+".png")
    if not title: fig.savefig("../figures/OLS_schedule12.png")
    plt.show()

def plot_mse_r2(PATH, filenames1, filenames2, title=None):

    n = len(filenames1)
    mse_sgd1 = np.zeros(n)
    mse_sgd2 = np.zeros(n)
    mse_basic1 = np.zeros(n)
    mse_basic2 = np.zeros(n)
    r2_sgd1 = np.zeros(n)
    r2_sgd2 = np.zeros(n)
    r2_basic1 = np.zeros(n)

    degrees = np.zeros(n)

    for i in range(n):
        name1 = filenames1[i]
        name2 = filenames2[i]
        sgd1, basic1 = get_data(PATH+name1)
        sgd2, basic2 = get_data(PATH+name2)

        mse_sgd1[i] = sgd1["mse"]
        mse_sgd2[i] = sgd2["mse"]
        mse_basic1[i] = basic1["mse"]

        r2_sgd1[i] = sgd1["r2"]
        r2_sgd2[i] = sgd2["r2"]
        r2_basic1[i] = basic1["r2"]

        degrees[i] = sgd1["d"]

    fig, ax = plt.subplots(nrows=2,
                           ncols=1,
                           sharex="col",
                           sharey=False,
                           figsize=[6.4, 6.4],
                           constrained_layout=True) #[6.4, 4.8].
    #ax[0].grid()
    ax[0].set_ylabel("MSE",fontsize = 10, fontname = "serif" )
    ax[0].plot(degrees, mse_sgd1, "tab:green", label = r"SGD $\alpha: $"+sgd1["lr"])
    ax[0].plot(degrees, mse_sgd2, "tab:blue", label = r"SGD $\alpha: $"+sgd2["lr"])
    ax[0].plot(degrees, mse_basic1, "tab:gray", linestyle="dashed", label = "Basic OLS")

    #ax[1].grid()
    ax[1].set_ylabel("R2",fontsize = 10, fontname = "serif" )
    ax[1].plot(degrees, r2_sgd1, "tab:green", label = r"SGD $\alpha: $"+sgd1["lr"])
    ax[1].plot(degrees, r2_sgd2, "tab:blue", label = r"SGD $\alpha: $"+sgd2["lr"])
    ax[1].plot(degrees, r2_basic1, "tab:gray", linestyle="dashed", label = "Basic OLS")

    lines = []
    labels = []

    for ax in fig.axes:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)

    fig.legend(lines[:3], labels[:3], loc = 'upper right')
    plt.xlabel("Model complexity (Degrees)")
    if title: fig.savefig("../figures/"+title+".png")
    if not title: fig.savefig("../figures/OLS_complexity.png")
    plt.show()

def plot_mom_epoch(PATH, filenames, title=None):

    fig = plt.figure()
    #plt.grid()

    for name in filenames:
        sgd, basic = get_data(PATH+name)
        n_epochs = len(sgd["cost"])
        epochs = np.arange(1, n_epochs+1)
        plt.plot(epochs, sgd["cost"], label=r"$\gamma$: "+sgd["gm"]+" (R2:{: .2f}, MSE:{: .2f})".format(sgd["r2"], sgd["mse"]))

    plt.xlabel("Epochs")
    plt.ylabel("Training loss")
    plt.legend()
    if title: fig.savefig("../figures/"+title+".png")
    if not title: fig.savefig("../figures/"+"SGDMOM_{:}_{:}_{:}_{:}_{:}.png".format(sgd["d"],sgd["ep"],sgd["bs"], sgd["gm"],len(filenames)))
    plt.show()

###############################################################################
def plot_mse_r2_ridge(PATH, filenames1, filenames2, title=None):

    n = len(filenames1)
    mse_sgd1 = np.zeros(n)
    mse_sgd2 = np.zeros(n)
    mse_basic1 = np.zeros(n)
    mse_basic2 = np.zeros(n)
    r2_sgd1 = np.zeros(n)
    r2_sgd2 = np.zeros(n)
    r2_basic1 = np.zeros(n)

    degrees = np.zeros(n)

    for i in range(n):
        name1 = filenames1[i]
        name2 = filenames2[i]
        sgd1, basic1 = get_data(PATH+name1)
        sgd2, basic2 = get_data(PATH+name2)

        mse_sgd1[i] = sgd1["mse"]
        mse_sgd2[i] = sgd2["mse"]
        mse_basic1[i] = basic1["mse"]

        r2_sgd1[i] = sgd1["r2"]
        r2_sgd2[i] = sgd2["r2"]
        r2_basic1[i] = basic1["r2"]

        degrees[i] = sgd1["d"]

    fig, ax = plt.subplots(nrows=2,
                           ncols=1,
                           sharex="col",
                           sharey=False,
                           figsize=[6.4, 6.4],
                           constrained_layout=True) #[6.4, 4.8].
    #ax[0].grid()
    ax[0].set_ylabel("MSE",fontsize = 10, fontname = "serif" )
    ax[0].plot(degrees, mse_sgd1, "tab:green", label = r"SGD $\alpha: $"+sgd1["lr"])
    ax[0].plot(degrees, mse_sgd2, "tab:blue", label = r"SGD $\alpha: $"+sgd2["lr"])
    ax[0].plot(degrees, mse_basic1, "tab:gray", linestyle="dashed", label = "Basic OLS")

    #ax[1].grid()
    ax[1].set_ylabel("R2",fontsize = 10, fontname = "serif" )
    ax[1].plot(degrees, r2_sgd1, "tab:green", label = r"SGD $\alpha: $"+sgd1["lr"])
    ax[1].plot(degrees, r2_sgd2, "tab:blue", label = r"SGD $\alpha: $"+sgd2["lr"])
    ax[1].plot(degrees, r2_basic1, "tab:gray", linestyle="dashed", label = "Basic OLS")

    lines = []
    labels = []

    for ax in fig.axes:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)

    fig.legend(lines[:3], labels[:3], loc = 'upper right')
    plt.xlabel("Model complexity (Degrees)")
    fig.savefig("../figures/{}.png".format(title))
    plt.show()


def plot_lamb_epoch(PATH, filenames, title=None):

    fig = plt.figure()

    for i in range(len(filenames)):
        name = filenames[i]
        sgd, basic = get_data(PATH+name)
        n_epochs = len(sgd["cost"])
        epochs = np.arange(1, n_epochs+1)
        label = r"$\$"
        label=r"$\lambda$: "+sgd["lamb"]+r"  $\alpha: $"+ sgd["lr"]+" (R2:{: .2f}, MSE:{: .2f})".format(sgd["r2"], sgd["mse"])
        plt.plot(epochs, sgd["cost"], label=label)

    plt.xlabel("Epochs")
    plt.ylabel("Training loss")
    plt.legend()
    fig.savefig("../figures/"+title+".png")
    plt.show()


def plot_lamb_epoch3(PATH, filenames1, filenames2, filenames3, title=None):

    fig, ax = plt.subplots(nrows=3,
                           ncols=1,
                           sharex="col",
                           sharey=False,
                           figsize=[6.4, 6.4],
                           constrained_layout=True) #[6.4, 4.8].

    for name in filenames1:
        sgd, basic = get_data(PATH+name)
        n_epochs = len(sgd["cost"])
        epochs = np.arange(1, n_epochs+1)
        label=r"$\lambda$: "+sgd["lamb"]+r"  $\alpha: $"+ sgd["lr"]+" (R2:{: .2f}, MSE:{: .2f})".format(sgd["r2"], sgd["mse"])
        ax[0].plot(epochs, sgd["cost"], label=label)
        ax[0].legend(loc = 'upper right')
        ax[0].set_ylabel("Training loss")

    for name in filenames2:
        sgd, basic = get_data(PATH+name)
        n_epochs = len(sgd["cost"])
        epochs = np.arange(1, n_epochs+1)
        label=r"$\lambda$: "+sgd["lamb"]+r"  $\alpha: $"+ sgd["lr"]+" (R2:{: .2f}, MSE:{: .2f})".format(sgd["r2"], sgd["mse"])
        ax[1].plot(epochs, sgd["cost"], label=label)
        ax[1].legend(loc = 'upper right')
        ax[1].set_ylabel("Training loss")
        #ax[1].semilogy()

    for name in filenames3:
        sgd, basic = get_data(PATH+name)
        n_epochs = len(sgd["cost"])
        epochs = np.arange(1, n_epochs+1)
        label=r"$\lambda$: "+sgd["lamb"]+r"  $\alpha: $"+ sgd["lr"]+" (R2:{: .2f}, MSE:{: .2f})".format(sgd["r2"], sgd["mse"])
        ax[2].plot(epochs, sgd["cost"], label=label)
        ax[2].legend(loc = 'upper right')
        ax[2].set_ylabel("Training loss")

    plt.xlabel("Epochs")
    fig.savefig("../figures/"+title+".png")
    plt.show()
##############################################################################
# OLS
###############################################################################
def main_LR(d, title=None):
    PATH = "../output/"

    filenames = [#"SGDLOG_OLS_{:}_100_5_0.5_standard_".format(d),
                 "SGDLOG_OLS_{:}_100_5_0.1_standard_".format(d),
                 "SGDLOG_OLS_{:}_100_5_0.05_standard_".format(d),
                 "SGDLOG_OLS_{:}_100_5_0.001_standard_".format(d),
                 "SGDLOG_OLS_{:}_100_5_0.0001_standard_".format(d),
                 #"SGDLOG_OLS_{:}_100_5_1e-05_standard_".format(d),
                 "SGDLOG_OLS_{:}_100_5_schedule_standard_".format(d)]

    plot_lr_epoch(PATH, filenames, title=title)

def main_LR_schedule(d, title=None):
    PATH = "../output/"

    filenames = ["SGDLOG_OLS_{:}_100_5_schedule2_standard_".format(d),
                 "SGDLOG_OLS_{:}_100_5_schedule_standard_".format(d)]

    plot_lr_epoch_shedule(PATH, filenames, title=title)

def main_LR_GM(d, title=None):
    PATH = "../output/"

    filenames1 = ["SGDLOG_OLS_{:}_100_5_0.1_standard_".format(d),
                 "SGDLOG_OLS_{:}_100_5_0.05_standard_".format(d)]

    filenames2 = ["SGDLOG_OLS_{:}_100_5_0.1_0.9_".format(d),
                  "SGDLOG_OLS_{:}_100_5_0.05_0.9_".format(d)]

    plot_lr_epoch3(PATH, filenames1, filenames2, title=title)

def main_complexity(lr, lr2, title=None):
    PATH = "../output/"

    filenames1 = ["SGDLOG_OLS_1_100_5_{:}_standard_".format(lr),
                 "SGDLOG_OLS_2_100_5_{:}_standard_".format(lr),
                 "SGDLOG_OLS_3_100_5_{:}_standard_".format(lr),
                 "SGDLOG_OLS_4_100_5_{:}_standard_".format(lr),
                 "SGDLOG_OLS_5_100_5_{:}_standard_".format(lr),
                 "SGDLOG_OLS_6_100_5_{:}_standard_".format(lr),
                 "SGDLOG_OLS_7_100_5_{:}_standard_".format(lr),
                 "SGDLOG_OLS_8_100_5_{:}_standard_".format(lr),
                 "SGDLOG_OLS_9_100_5_{:}_standard_".format(lr)]

    filenames2 = ["SGDLOG_OLS_1_100_5_{:}_standard_".format(lr2),
                 "SGDLOG_OLS_2_100_5_{:}_standard_".format(lr2),
                 "SGDLOG_OLS_3_100_5_{:}_standard_".format(lr2),
                 "SGDLOG_OLS_4_100_5_{:}_standard_".format(lr2),
                 "SGDLOG_OLS_5_100_5_{:}_standard_".format(lr2),
                 "SGDLOG_OLS_6_100_5_{:}_standard_".format(lr2),
                 "SGDLOG_OLS_7_100_5_{:}_standard_".format(lr2),
                 "SGDLOG_OLS_8_100_5_{:}_standard_".format(lr2),
                 "SGDLOG_OLS_9_100_5_{:}_standard_".format(lr2)]

    plot_mse_r2(PATH, filenames1, filenames2, title=title)

def main_momentum(d, title=None):
    PATH = "../output/"

    filenames = ["SGDLOG_OLS_{:}_100_5_0.1_0.9_".format(d),
                 "SGDLOG_OLS_{:}_100_5_0.1_0.5_".format(d),
                 "SGDLOG_OLS_{:}_100_5_0.1_0.1_".format(d),
                 "SGDLOG_OLS_{:}_100_5_0.1_0.01_".format(d)]

    plot_mom_epoch(PATH, filenames, title=title)

###############################################################################
#     RIDGE
###############################################################################
def main_lamb(d, title=None):
    PATH = "../output/"

    filenames1 = ["SGDLOG_Ridge_0.1_{:}_100_5_0.1_standard_".format(d),
                 "SGDLOG_Ridge_0.001_{:}_100_5_0.1_standard_".format(d),
                 "SGDLOG_Ridge_1e-05_{:}_100_5_0.1_standard_".format(d)]

    filenames2 = ["SGDLOG_Ridge_0.1_{:}_100_5_0.05_standard_".format(d),
                 "SGDLOG_Ridge_0.001_{:}_100_5_0.05_standard_".format(d),
                 "SGDLOG_Ridge_1e-05_{:}_100_5_0.05_standard_".format(d)]

    filenames3 = ["SGDLOG_Ridge_0.1_{:}_100_5_schedule_standard_".format(d),
                 "SGDLOG_Ridge_0.001_{:}_100_5_schedule_standard_".format(d),
                 "SGDLOG_Ridge_1e-05_{:}_100_5_schedule_standard_".format(d)]

    plot_lamb_epoch3(PATH, filenames1, filenames2, filenames3, title="Ridge_loss")


##############################################################################
def main_OLS():
    main_LR(d=7, title="OLS_all_lr")
    main_LR_schedule(d=7, title="OLS_shedule_lr")
    main_LR_GM(d=7, title="OLS_learn_momentum")
    main_complexity(lr = 0.1, lr2 = 0.05, title="OLS_complexity")
    main_momentum(d=7, title= "OLS_momentum")

def main_Ridge():
    main_lamb(d=7, title="ridge_2")

if __name__ == '__main__':
    main_OLS()
    main_Ridge()
