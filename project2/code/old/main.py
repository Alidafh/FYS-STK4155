#!/usr/bin/python
import numpy as np
import os

def run_learn_rate(r="OLS", lamb=0.001, d=4, ep=100, bs=5):
    learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    for lr in learning_rates:
        options = "-r {:} -l {:} -d {:} -ep {:} -bs {:} -lr {:}".format(r, lamb, d, ep, bs, lr)
        os.system("python gradient.py "+options)

    options = "-r {:} -l {:} -d {:} -ep {:} -bs {:}".format(r, lamb, d, ep, bs)
    os.system("python gradient.py "+options)


def run_degree(d_max, r="OLS", lamb=0.001, ep=100, bs=5, lr=None):
    degrees = np.arange(1, d_max+1, dtype=int)
    for d in degrees:
        options = "-r {:} -l {:} -d {:} -ep {:} -bs {:}".format(r, lamb, d, ep, bs)
        if lr:
            options = "-r {:} -l {:} -d {:} -ep {:} -bs {:} -lr {:}".format(r, lamb, d, ep, bs, lr)
        os.system("python gradient.py "+options)

def run_schedule_bs(r="OLS", lamb=0.001, d=4, ep=100, bs = 5, lr=None):
    batch_sizes = [1, 3, 5, 7, 10]
    for bs in batch_sizes:
        options = "-r {:} -l {:} -d {:} -ep {:} -bs {:}".format(r, lamb, d, ep, bs)
        if lr:
            options = "-r {:} -l {:} -d {:} -ep {:} -bs {:} -lr {:}".format(r, lamb, d, ep, bs, lr)
        os.system("python gradient.py "+ options)


if __name__ == '__main__':
    run_learn_rate(d=4)
    run_degree(d_max=10, lr=0.01)
    #run_degree(d_max=10)
    #run_degree(d_max=10, lr=0.001)
