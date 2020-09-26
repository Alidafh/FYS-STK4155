#!/usr/bin/python
import matplotlib.pyplot as plt
from matplotlib import cm
import functions as func
import numpy as np

def plot_franke(title, filename):
    """
    Plots the franke function and saves it in the folder: output/figures/
    --------------------------------
    Input
        title: title of the figure
        filename: the name of the saved file
    --------------------------------
    TODO: change back to saving in PDF format
    """
    print("Plotting an illustration of the franke function")
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x,y)

    z = func.FrankeFunction(x,y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x,y,z,cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.title(title, fontsize = 12, fontname = "serif")
    ax.set_xlabel(r"$x$", fontsize = 12, fontname = "serif")
    ax.set_ylabel(r"$y$", fontsize = 12, fontname = "serif")
    ax.set_zlabel(r"$z$", fontsize = 12, fontname = "serif")
    #plt.savefig("output/figures/{}.pdf".format(filename), scale=0.1)
    plt.savefig("output/figures/{}.png".format(filename), scale=0.1)
    print("    Figure saved in: output/figures/{}.pdf\n".format(filename))
    #plt.show()

def plot_MSE(x, test_error, train_error, rType = "", c=""):
    """
    Plots the Mean Squared Error as a function of model complexity, saves figure in output/figures/
    --------------------------------
    Input
        x: the model complexity
        test_error/train_error: the mean squared error of test/train set
        rType: regression type (OLS, RIDGE, LASSO)
        c: complexity
    --------------------------------
    TODO: change back to saving in PDF format
    """
    print("Plotting the MSE of the training and test results")
    fig = plt.figure()
    plt.grid()
    plt.title("{} Mean Squared Errors".format(rType), fontsize = 12, fontname = "serif")
    plt.plot(x, test_error, "tab:green", label="Test Error")
    plt.plot(x, train_error,"tab:blue", label="Train Error")
    plt.semilogy()
    plt.legend()
    plt.xlabel("Model complexity ({})".format(c.split("_")[0]), fontsize = 12, fontname = "serif")
    plt.ylabel("MSE", fontsize = 12, fontname = "serif")
    plt.savefig("output/figures/MSE_{}_{}.png".format(rType, c))
    print("    Figure saved in: output/figures/MSE_{}_{}.png\n".format(rType, c))
    #plt.show()

def bias_variance(x, mse, var, bias, rType = "", c=""):
    """
    Plots the Bias, Variance and MSE as a function of degrees
    --------------------------------
    Input
        x: the model complexity
        mse:
        var:
        bias:
        rType: regression type (OLS, RIDGE, LASSO)
        c: complexity type
    --------------------------------
    TODO: change back to saving in PDF format
    """
    print("Plotting the Bias, Variance and MSE")
    fig = plt.figure()
    plt.grid()
    plt.title("{} Bias-Variance".format(rType), fontsize = 12, fontname = "serif")
    plt.plot(x, bias, "tab:green", label="Bias")
    plt.plot(x, var, "tab:blue", label="Variance")
    plt.plot(x, mse, "tab:red", label="MSE")
    plt.legend()
    plt.xlabel("Model complexity ({})".format(c.split("_")[0]), fontsize = 12, fontname = "serif")
    plt.savefig("output/figures/bias_variance_{}_{}.png".format(rType, c))
    print("    Figure saved in: output/figures/bias_variance_{}_{}.png\n".format(rType, c))

if __name__ == '__main__':
    #plot_franke("Illustration of the Franke Function", "franke_func_illustration")
    type = "OLS_degree5"
    print(type.split("_")[0])
