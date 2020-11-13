#!/usr/bin/python
import numpy as np
import os

def run_learn_rate(r="OLS", lamb=0.001, d=4, ep=100, bs=5, p="False"):
    learning_rates = [0.5, 0.1, 0.05, 0.001, 0.0001]
    for lr in learning_rates:
        options = "-r {:} -l {:} -d {:} -ep {:} -bs {:} -lr {:} -p {:}".format(r, lamb, d, ep, bs, lr, p)
        os.system("python gradient.py "+options)

    options = "-r {:} -l {:} -d {:} -ep {:} -bs {:} -p{:}".format(r, lamb, d, ep, bs, p)
    os.system("python gradient.py "+options)

def run_learn_rate2(r="OLS", lamb=0.001, d=4, ep=100, bs=5, p="False", gm=None):
    learning_rates = [0.5, 0.1, 0.05, 0.01, 0.001, 0.0001]

    for lr in learning_rates:

        options = "-r {:} -l {:} -d {:} -ep {:} -bs {:} -lr {:} -p {:}".format(r, lamb, d, ep, bs, lr, p)
        if gm:
            options = "-r {:} -l {:} -d {:} -ep {:} -bs {:} -lr {:} -p {:} -gm {:}".format(r, lamb, d, ep, bs, lr, p, gm)

        os.system("python gradient.py "+options)

def run_momentum(r="OLS", lamb=0.001, d=4, ep=100, bs=5, p="False", lr=None):
    gms = [0.9, 0.5, 0.1, 0.01]
    for gm in gms:
        options = "-r {:} -l {:} -d {:} -ep {:} -bs {:} -p {:} -gm {:}".format(r, lamb, d, ep, bs, p, gm)
        if lr:
            options = "-r {:} -l {:} -d {:} -ep {:} -bs {:} -lr {:} -p {:} -gm {:}".format(r, lamb, d, ep, bs, lr, p, gm)
        os.system("python gradient.py "+options)

def run_degree(d_max, r="OLS", lamb=0.001, ep=100, bs=5, lr=None, p="False"):
    degrees = np.arange(1, d_max+1, dtype=int)
    for d in degrees:
        options = "-r {:} -l {:} -d {:} -ep {:} -bs {:} -p {:}".format(r, lamb, d, ep, bs, p)
        if lr:
            options = "-r {:} -l {:} -d {:} -ep {:} -bs {:} -lr {:} -p {:}".format(r, lamb, d, ep, bs, lr, p)
        os.system("python gradient.py "+options)

def run_bs(r="OLS", lamb=0.001, d=4, ep=100, bs = 5, lr=None, p="False"):
    batch_sizes = [2, 5, 10, 100]
    for bs in batch_sizes:
        options = "-r {:} -l {:} -d {:} -ep {:} -bs {:} -lr {:} -p {:}".format(r, lamb, d, ep, bs, lr, p)
        if lr:
            options = "-r {:} -l {:} -d {:} -ep {:} -bs {:} -lr {:} -p {:}".format(r, lamb, d, ep, bs, lr, p)
        os.system("python gradient.py "+ options)


def run_lamb(r="Ridge", d=4, ep=100, bs=5, p="False", lr=None):
    lambs = [0.1, 0.001, 0.0001, 0.00001]
    for lamb in lambs:
        options = "-r {:} -l {:} -d {:} -ep {:} -bs {:} -p {:}".format(r, lamb, d, ep, bs, p)
        if lr:
            options = "-r {:} -l {:} -d {:} -ep {:} -bs {:} -lr {:} -p {:}".format(r, lamb, d, ep, bs, lr, p)
        os.system("python gradient.py "+ options)

if __name__ == '__main__':
    #run_degree(r="Ridge", d_max=10, lr=0.1, lamb=0.1)
    run_lamb(d=7, lr=0.1)
    run_lamb(d=7, lr=0.05)
    run_lamb(d=7)
