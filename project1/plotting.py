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

def OLS_test_train(x, test_error, train_error, err_type ="", info="", log=False):
    """
    Plots the Mean Squared Error as a function of model complexity, saves figure in output/figures/
    --------------------------------
    Input
        x: the model complexity
        test_error/train_error: the mean squared error of test/train set
        errr_type: MSE, var etc.
    --------------------------------
    TODO: change back to saving in PDF format
    """

    print("Plotting the {} of the training and test results".format(err_type))
    fig = plt.figure()
    plt.grid()

    plt.title("{} Mean Squared Errors".format(err_type), fontsize = 12, fontname = "serif")
    plt.xlabel("Model complexity (degrees)", fontsize = 12, fontname = "serif")
    plt.ylabel("{}".format(err_type), fontsize = 12, fontname = "serif")

    plt.plot(x, test_error, "tab:green", label="Test Error")
    plt.plot(x, train_error,"tab:blue", label="Train Error")
    plt.legend()

    if log==True:plt.semilogy()

    fig.savefig("output/figures/OLS_{:}_test_train_{:}.png".format(err_type, info))
    fig.savefig("output/figures/OLS_{:}_test_train_{:}.png".format(err_type, info))
    print("    Figure saved in: output/figures/OLS_{:}_test_train_{:}.pdf\n".format(err_type, info))
    plt.close()

def OLS_bias_variance(x, mse, var, bias, info, log=False):
    """
    Plots the Bias, Variance and MSE as a function of degrees
    --------------------------------
    Input
        x: the model complexity
        mse:
        var:
        bias:
        info: info for the filename (number of datapoints etc.)
    --------------------------------
    TODO: change back to saving in PDF format
    """
    print("Plotting the Bias, Variance and MSE")
    fig = plt.figure()
    plt.grid()
    plt.title("OLS Bias-Variance", fontsize = 12, fontname = "serif")
    plt.xlabel("Model complexity (degrees)", fontsize = 12, fontname = "serif")

    plt.plot(x, mse, "tab:red", label="MSE")
    plt.plot(x, var, "tab:blue", label="Variance")
    plt.plot(x, bias, "tab:green", label="Bias")
    plt.legend()
    if log==True: plt.semilogy()

    #fig.savefig("output/figures/OLS_bias_variance_{}.pdf".format(info))
    fig.savefig("output/figures/OLS_bias_variance_{}.png".format(info))
    print("    Figure saved in: output/figures/OLS_bias_variance_{}.pdf\n".format(info))
    #plt.close()

def OLS_beta_conf(beta, conf_beta, d):
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

    print("Plotting the OLS regression parameters with confidence intervals")
    fig, ax = plt.subplots()
    ax.grid()
    plt.title("OLS parameters using polynomial degree {:.0f} ".format(d), fontsize = 12, fontname = "serif")
    x = np.arange(len(beta))
    plt.errorbar(x, beta, yerr=conf_beta, markersize=4, linewidth=1, capsize=5, capthick=1, ecolor="black", fmt='o')
    xlabels = [r"$\beta_"+"{:.0f}$".format(i) for i in range(len(beta))]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)

    #fig.savefig("output/figures/OLS_parameters_degree_{:.0f}.png".format(d))
    fig.savefig("output/figures/OLS_parameters_degree_{:.0f}.png".format(d))
    print("    Figure saved in: output/figures/OLS_beta_degree_{:.0f}.pdf\n".format(d))
    plt.close()

def OLS_allfolds(x, var, k, rType="", varN="", log=False):
    """
    Plots the chosen variable var as a function of degrees for all k folds
    used in k-fold. Saves the image in output/figures
    --------------------------------
    Input
        x: degrees
        var: the variable you want to plot
        k: number of folds
        rType: string, regression type
        varN: string, name of variable
    --------------------------------
    TODO: change back to saving in PDF format
    """
    print("Plotting the {:} of all {:.0f} folds".format(varN,k))
    fig = plt.figure()
    plt.grid()
    plt.title("{:} {:} (k={:.0f})".format(rType, varN, k), fontsize = 12, fontname = "serif")
    label = ["k={}".format(i) for i in range(1, k+1)]
    plt.plot(x, var)
    if log==True: plt.semilogy()
    plt.legend(label)
    plt.xlabel("Model complexity (degrees)", fontsize = 12, fontname = "serif")
    plt.ylabel("{}".format(varN), fontsize = 12, fontname = "serif")

    #fig.savefig("output/figures/OLS_allfolds_{:.0f}_{:}_{:}.pdf".format(k, rType, varN))
    fig.savefig("output/figures/OLS_allfolds_{:.0f}_{:}_{:}.png".format(k, rType, varN))
    print("    Figure saved in: output/figures/OLS_allfolds_{:.0f}_{:}_{:}.pdf\n".format(k, rType, varN))
    plt.close()

def OLS_metric(x, var, varN="", info="", log=False):
    """
    Plots the chosen variable as a function of degrees
    --------------------------------
    Input
        x: degrees
        var: the variable you want to plot
        varN: string, name of variable
        info: information for filename
    --------------------------------
    TODO: change back to saving in PDF format
    """
    print("Plotting the {:}".format(varN))
    fig = plt.figure()
    plt.grid()
    plt.title("OLS {:}".format(varN), fontsize = 12, fontname = "serif")
    plt.xlabel("Model complexity (degrees)", fontsize = 12, fontname = "serif")
    plt.ylabel("{}".format(varN), fontsize = 12, fontname = "serif")

    plt.plot(x, var, label="{}".format(varN))
    plt.legend()
    if log==True: plt.semilogy()

    #fig.savefig("output/figures/OLS_{:}_{:}.png".format(varN, info))
    fig.savefig("output/figures/OLS_metric_{:}_{:}.png".format(varN, info))
    print("    Figure saved in: output/figures/OLS_{:}_{:}.pdf\n".format(varN, info))
    plt.close()


def kFold_all_metrics(x, est_metrics, info):
    """
    Plots the metrics found using k-fold
    --------------------------------
    Input
        x: degrees
        vars: the est metrics
        info: string of info for filename (type of regression etc.)
    --------------------------------
    TODO: not done
    """
    print("Plotting the estimated MSE, Variance, and Bias from kFold")

    fig = plt.figure()
    plt.grid()
    plt.title("MSE, Bias and variance using kFold", fontsize = 12, fontname = "serif")

    plt.plot(x, est_metrics)
    plt.legend(["MSE", "Variance", "Bias"])
    plt.xlabel("Model complexity (degrees)", fontsize = 12, fontname = "serif")

    #fig.savefig("output/figures/kFold_bias_variance_{}.pdf".format(info))
    fig.savefig("output/figures/kFold_bias_variance_{}.png".format(info))
    print("    Figure saved in: output/figures/kFold_bias_variance_{}\n.pdf".format(info))

    plt.close()


if __name__ == '__main__':
    #plot_franke("Illustration of the Franke Function", "franke_func_illustration")
    type = "OLS_degree5"
    print(type.split("_")[0])
