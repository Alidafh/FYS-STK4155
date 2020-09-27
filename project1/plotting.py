#!/usr/bin/python
import matplotlib.pyplot as plt
from matplotlib import cm
import functions as func
import numpy as np

def plot_franke(title, filename, noise=0):
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

    z = func.FrankeFunction(x,y) + np.random.normal(0, noise, len(x))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x,y,z,cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.title(title, fontsize = 12, fontname = "serif")
    ax.set_xlabel(r"$x$", fontsize = 12, fontname = "serif")
    ax.set_ylabel(r"$y$", fontsize = 12, fontname = "serif")
    ax.set_zlabel(r"$z$", fontsize = 12, fontname = "serif")
    #plt.savefig("output/figures/{}.pdf".format(filename))
    fig.savefig("output/figures/{:}_{:}.png".format(filename, noise), scale=0.1)
    print("    Figure saved in: output/figures/{:}_{:}.pdf\n".format(filename, noise))
    plt.close()

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
    #plt.savefig("output/figures/MSE_{}_{}.pdf".format(rType, c))
    fig.savefig("output/figures/MSE_{}_{}.png".format(rType, c))
    print("    Figure saved in: output/figures/MSE_{}_{}.pdf\n".format(rType, c))
    plt.close()

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
    plt.semilogy()
    plt.xlabel("Model complexity ({})".format(c.split("_")[0]), fontsize = 12, fontname = "serif")
    #fig.savefig("output/figures/bias_variance_{}_{}.pdf".format(rType, c))
    fig.savefig("output/figures/bias_variance_{}_{}.png".format(rType, c))
    print("    Figure saved in: output/figures/bias_variance_{}_{}.pdf\n".format(rType, c))
    plt.close()


def plot_beta(beta, conf_beta, d):
    """
    Plots the parameters with errorbars corresponding to the confidence interval
    --------------------------------
    Input
        beta: the regression parameters
        conf_beta: the std of the regression parameters
        d: the polynomial degree
    --------------------------------
    TODO: change back to saving in PDF format
    """

    print("Plotting the regression parameters with confidence intervals")
    fig, ax = plt.subplots()
    ax.grid()
    plt.title("OLS parameters using polynomial degree {:.0f} ".format(d), fontsize = 12, fontname = "serif")
    x = np.arange(len(beta))
    plt.errorbar(x, beta, yerr=conf_beta, markersize=4, linewidth=1, capsize=5, capthick=1, ecolor="black", fmt='o')
    xlabels = [r"$\beta_"+"{:.0f}$".format(i) for i in range(len(beta))]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    #fig.savefig("output/figures/beta_degree{:.0f}.pdf".format(d))
    fig.savefig("output/figures/beta_degree_{:.0f}.png".format(d))
    print("    Figure saved in: output/figures/beta_degree_{:.0f}.pdf\n".format(d))
    plt.close()

def plot_kFold_var(x, var, k, rType="", varN=""):
    print("Plotting the MSE with {:.0f} folds".format(k))
    fig = plt.figure()
    plt.grid()
    plt.title("{:} {:} (k={:.0f})".format(rType, varN, k), fontsize = 12, fontname = "serif")
    label = ["k={}".format(i) for i in range(1, k+1)]
    plt.plot(x, var)
    #plt.semilogy()
    plt.legend(label)
    plt.xlabel("Model complexity (degrees)", fontsize = 12, fontname = "serif")
    plt.ylabel("{}".format(varN), fontsize = 12, fontname = "serif")
    #fig.savefig("output/figures/bias_variance_{}_{}.pdf".format(rType, c))
    fig.savefig("output/figures/{:}_{:}_k{:.0f}.png".format(rType, varN, k))
    print("    Figure saved in: output/figures/{:}_{:}_k{:.0f}.png".format(rType, varN, k))
    plt.close()


if __name__ == '__main__':
    #plot_franke("Illustration of the Franke Function", "franke_func_illustration")
    type = "OLS_degree5"
    print(type.split("_")[0])
