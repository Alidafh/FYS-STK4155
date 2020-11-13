#!/usr/bin/python
import numpy as np
import tools as tools
from sklearn.model_selection import train_test_split
from regression import OLS, Ridge
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy
from tabulate import tabulate


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


learning_rates = [0.1, 0.05, 0.001, 0.0001]
batch_sizes = [2, 5, 10, 100]
PATH="output/"
"""
f = []
for lr in learning_rates:
    for bs in batch_sizes:
        name = "SGDLOG_OLS_7_100_{:}_{:}_standard_".format(bs, lr)
        print(name)
        f.append(name)
print(f)
quit()
"""
filenames =['SGDLOG_OLS_7_100_2_0.1_standard_',
            'SGDLOG_OLS_7_100_5_0.1_standard_',
            'SGDLOG_OLS_7_100_10_0.1_standard_',
            'SGDLOG_OLS_7_100_100_0.1_standard_',
            'SGDLOG_OLS_7_100_2_0.05_standard_',
            'SGDLOG_OLS_7_100_5_0.05_standard_',
            'SGDLOG_OLS_7_100_10_0.05_standard_',
            'SGDLOG_OLS_7_100_100_0.05_standard_',
            'SGDLOG_OLS_7_100_2_0.001_standard_',
            'SGDLOG_OLS_7_100_5_0.001_standard_',
            'SGDLOG_OLS_7_100_10_0.001_standard_',
            'SGDLOG_OLS_7_100_100_0.001_standard_',
            'SGDLOG_OLS_7_100_2_0.0001_standard_',
            'SGDLOG_OLS_7_100_5_0.0001_standard_',
            'SGDLOG_OLS_7_100_10_0.0001_standard_',
            'SGDLOG_OLS_7_100_100_0.0001_standard_']




for i in range(len(filenames)):
    name = filenames[i]
    sgd, basic = get_data(PATH+name)
    print("BS: {:}\t | LR: {:}\t | R2: {: .2f} \t | MSE: {: .2f}".format(sgd["bs"], sgd["lr"], sgd["r2"], sgd["mse"]))
