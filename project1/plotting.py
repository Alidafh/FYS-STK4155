#!/usr/bin/python
import matplotlib.pyplot as plt
from matplotlib import cm
import functions as func
import numpy as np
import os, errno

plt.rc('font',family='serif')
###########


def OLS_bias_variance(x, mse, var, bias, x_type="degrees", info="", log=False):
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
    plt.title("OLS Bias-Variance", fontsize = 14, fontname = "serif")
    plt.xlabel("Model complexity (degrees)", fontsize = 12, fontname = "serif")

    plt.plot(x, mse, "tab:red", label="MSE")
    plt.plot(x, var, "tab:blue", label="Variance")
    plt.plot(x, bias, "tab:green", label="Bias")
    plt.legend()
    if log==True: plt.semilogy()

    fig.savefig("output/figures/OLS_bias_variance_{:}_{:}.pdf".format(x_type, info))
    fig.savefig("output/figures/OLS_bias_variance_{:}_{:}.png".format(x_type, info))
    print("    Figure saved in: output/figures/OLS_bias_variance_{:}_{:}.pdf\n".format(x_type, info))
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
    plt.title("OLS {:}".format(varN), fontsize = 14, fontname = "serif")
    plt.xlabel("Model complexity (degrees)", fontsize = 12, fontname = "serif")
    plt.ylabel("{}".format(varN), fontsize = 12, fontname = "serif")

    plt.plot(x, var, label="{}".format(varN))
    plt.legend()
    if log==True: plt.semilogy()

    #fig.savefig("output/figures/OLS_{:}_{:}.png".format(varN, info))
    fig.savefig("output/figures/OLS_metric_{:}_{:}.png".format(varN, info))
    print("    Figure saved in: output/figures/OLS_{:}_{:}.pdf\n".format(varN, info))
    plt.close()

def RIDGE_test_train(x, test_error, train_error, lamb, err_type ="", info="", log=False):
    """
    Plots the Mean Squared Error as a function of model complexity,
    saves figure in output/figures/
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

    plt.title(r"Ridge {:}, $\lambda={:.4f}$".format(err_type, lamb), fontsize = 14, fontname = "serif")
    plt.xlabel("Model complexity (degrees)", fontsize = 12, fontname = "serif")
    plt.ylabel("{}".format(err_type), fontsize = 12, fontname = "serif")

    plt.plot(x, test_error, "tab:green", label="Test Error")
    plt.plot(x, train_error,"tab:blue", label="Train Error")
    plt.legend()

    if log==True: plt.semilogy()

    #fig.savefig("output/figures/RIDGE_{:}_test_train_{:}.pdf".format(err_type, info))
    fig.savefig("output/figures/RIDGE_{:}_test_train_{:}_lambd{:}.png".format(err_type, info, lamb))
    print("    Figure saved in: output/figures/RIDGE_{:}_test_train_{:}_lambd{:}.pdf\n".format(err_type, info, lamb))
    plt.close()


###############################################################################
#            Functions that work for both OLS and RIDGE
###############################################################################

def allfolds(x, var, k, n, rType="", varN="", log=False, lamb=0):
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
    plt.title("{:} {:} (k={:.0f})".format(rType, varN, k), fontsize = 14, fontname = "serif")
    if rType=="RIDGE":
        plt.title("{:} {:} (k={:.0f}, $\lambda={:.6f}$)".format(rType, varN, k, lamb), fontsize = 14, fontname = "serif")
    label = ["k={}".format(i) for i in range(1, k+1)]
    plt.plot(x, var)
    if log==True: plt.semilogy()
    plt.legend(label)
    plt.xlabel("Model complexity (degrees)", fontsize = 12, fontname = "serif")
    plt.ylabel("{}".format(varN), fontsize = 12, fontname = "serif")

    fig.savefig("output/figures/{:}_allfolds_k{:.0f}_ndata{:.0f}_{:}.png".format(rType, k, n, varN))
    print("    Figure saved in: output/figures/{:}_allfolds_k{:.0f}_ndata{:.0f}_{:}.pdf\n".format(rType, k, n, varN))
    plt.close()

def metric_test_train(x, test_var, train_var, var_type ="", x_type="", reg_type="", info="", log=False):
    """
    Plots the chosen metric as a function of x for both test and training set
    saves figure in output/figures/
    --------------------------------
    Input
        x:
        test_var/train_var: the variable to plot
        var_type: MSE, variance, Bias, R2 var etc.
        x_type: what you are plotting against
    --------------------------------
    TODO: change back to saving in PDF format
    """

    print("Plotting the {} results".format(var_type))

    fig = plt.figure()
    plt.grid()

    titles = ["Explained R2-score", "Mean Squared Error", "Variance", "Bias"]
    if var_type == "R2": plt.title(reg_type+ " " + titles[0], fontsize = 14, fontname = "serif")
    if var_type == "MSE": plt.title(reg_type+ " " + titles[1], fontsize = 14, fontname = "serif")
    if var_type == "VAR": plt.title(reg_type+ " " + titles[2], fontsize = 14, fontname = "serif")
    if var_type == "BIAS": plt.title(reg_type+ " " + titles[3], fontsize = 14, fontname = "serif")

    x_label = "Model Complexity (Degrees)"
    if x_type == "data": x_label = "Size of dataset"

    plt.xlabel(x_label, fontsize = 12, fontname = "serif")
    plt.ylabel("{}".format(var_type), fontsize = 12, fontname = "serif")

    plt.plot(x, test_var, "tab:green", label="Test")
    plt.plot(x, train_var,"tab:blue", label="Train")
    plt.legend()

    if log==True: plt.semilogy()

    #fig.savefig("output/figures/{:}_{:}_test_train_{:}.pdf".format(reg_type, var_type, info))
    fig.savefig("output/figures/{:}_{:}_test_train_{:}.png".format(reg_type, var_type, info))
    print("    Figure saved in: output/figures/{:}_{:}_test_train_{:}.png\n".format(reg_type, var_type, info))
    plt.close()


###############################################################################
#            Functions in use
###############################################################################
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
    plt.title(title, fontsize = 14, fontname = "serif")
    ax.set_xlabel(r"$x$", fontsize = 12, fontname = "serif")
    ax.set_ylabel(r"$y$", fontsize = 12, fontname = "serif")
    ax.set_zlabel(r"$z$", fontsize = 12, fontname = "serif")
    #plt.savefig("output/figures/{}.pdf".format(filename))

    if not os.path.exists(os.path.dirname("output/figures/")):
        try:
            os.makedirs(os.path.dirname("output/figures/"))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    fig.savefig("output/figures/{:}_{:}.pdf".format(filename, noise), scale=0.1)
    print("    Figure saved in: output/figures/{:}_{:}.pdf\n".format(filename, noise))
    plt.close()

def OLS_test_train(x, test_error, train_error, err_type ="", info="", log=False):
    """
    Plots the Mean Squared Error as a function of model complexity,
    saves figure in output/figures/
    --------------------------------
    Input
        x: the model complexity
        test_error/train_error: the mean squared error of test/train set
        errr_type: MSE, var etc.
    --------------------------------
    TODO: change back to saving in PDF format
    """

    print("Plotting the {} of the training and test results(2.11 Hastie)".format(err_type))
    fig = plt.figure()
    plt.grid()

    plt.title("OLS Mean Squared Errors", fontsize = 14, fontname = "serif")
    plt.xlabel("Model complexity (degrees)", fontsize = 12, fontname = "serif")
    plt.ylabel("{}".format(err_type), fontsize = 12, fontname = "serif")

    plt.plot(x, test_error, "tab:green", label="Test Error")
    plt.plot(x, train_error,"tab:blue", label="Train Error")
    plt.legend()

    if log==True: plt.semilogy()

    fig.savefig("output/figures/OLS_{:}_test_train_{:}.pdf".format(err_type, info))
    fig.savefig("output/figures/OLS_{:}_test_train_{:}.png".format(err_type, info))
    print("    Figure saved in: output/figures/OLS_{:}_test_train_{:}.pdf\n".format(err_type, info))
    plt.close()

def compare_R2(x, r2, r2_bs, r2_k, rType = "OLS", lamb=0, info=""):
    """
    Compares the test and train R2-scores for no resampling, bootstrap and
    k-fold methods as a function of model complexity
    --------------------------------
    Input
    --------------------------------
    TODO: change back to saving in PDF format
    """

    print("Comparing R-score for k-fold and bootstrap methods")
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex="col", sharey=False, figsize=[6.4, 6.4], constrained_layout=True) #[6.4, 4.8].
    fig.suptitle("{:} R2-score".format(rType), fontsize = 14, fontname = "serif")
    if rType != "OLS":
        fig.suptitle("{:} R2-score (\lambda = {:.5f})".format(rType, lamb), fontsize = 14, fontname = "serif")

    labels = ["R2-test", "R2-train"]
    for i in range(2):
        ax[i].grid()
        ax[i].plot(x, r2[i], color="tab:green", label="No-resampling")
        ax[i].plot(x, r2_bs[i], color="tab:blue", label="Bootstrap")
        ax[i].plot(x, r2_k[i], color="tab:red", label="k-Fold")
        ax[i].set_ylabel(labels[i],fontsize = 10, fontname = "serif" )
        if len(x) <= 10: ax[i].set_xticks(x)

    lines = []
    labels = []

    for ax in fig.axes:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)

    fig.legend(lines[:3], labels[:3], loc = 'lower right')
    plt.xlabel("Model complexity (Degrees)")

    fig.savefig("output/figures/{:}_compare_R2_{:}.pdf".format(rType, info))
    fig.savefig("output/figures/{:}_compare_R2_{:}.png".format(rType, info))
    print("    Figure saved in: output/figures/{:}_compare_R2_{:}.pdf\n".format(rType, info))
    plt.close()

def compare_MSE(x, mse, mse_bs, mse_k, rType = "OLS", lamb=0, info="", log=False):
    """
    Compares the MSE for no resampling, bootstrap and
    k-fold methods as a function of model complexity
    --------------------------------
    Input
    --------------------------------
    TODO: change back to saving in PDF format
    """

    print("Comparing MSE for k-fold and bootstrap methods")
    fig = plt.figure()
    plt.grid()
    plt.title("{:} Mean Squared Error".format(rType), fontsize = 14, fontname = "serif")
    if rType != "OLS":
        plt.title("{:} Mean Squared Error (\lambda={:.6f})".format(rType, lamb), fontsize = 14, fontname = "serif")

    plt.xlabel("Model complexity (degrees)", fontsize = 12, fontname = "serif")
    plt.ylabel("MSE", fontsize=12, fontname="serif")
    plt.plot(x, mse, "tab:red", label="no resample")
    plt.plot(x, mse_bs, "tab:blue", label="bootstrap")
    plt.plot(x, mse_k, "tab:green", label="k-Fold")
    plt.legend()

    if log==True: plt.semilogy()

    fig.savefig("output/figures/{:}_compare_MSE_{:}.pdf".format(rType, info))
    fig.savefig("output/figures/{:}_compare_MSE_{:}.png".format(rType, info))
    print("    Figure saved in: output/figures/{:}_compare_MSE_{:}.pdf\n".format(rType, info))
    #plt.show()
    plt.close()

def bias_variance_m(x, m1, m2, m3, d1, d2, d3, x_type="degrees", RegType ="OLS", info="", log=False):
    """
    """
    print("Plotting the Bias, Variance and MSE for {:} for multiple degrees".format(RegType))

    fig = plt.figure()
    plt.grid()
    plt.title("{:} Bias-Variance, d = {:}, {:}, {:}".format(RegType, d1, d2, d3), fontsize = 14, fontname = "serif")

    if x_type == "degrees": plt.xlabel("Model complexity (degrees)", fontsize = 12, fontname = "serif")
    if x_type == "data": plt.xlabel("Number of datapoints (n)", fontsize = 12, fontname = "serif")
    if x_type == "lambda": plt.xlabel("$\lambda$ (log scale)", fontsize = 12, fontname = "serif")
    met = [m1, m2, m3]
    deg = [d1, d2, d3]
    lines = ["solid", "dotted", "dashed"]
    for i in range(3):
        plt.plot(x, met[i][1], "tab:red", linestyle=lines[i], label="MSE (d={:.0f})".format(deg[i]))
        plt.plot(x, met[i][2], "tab:red", linestyle=lines[i], label="Variance (d={:.0f})".format(deg[i]))
        plt.plot(x, met[i][3], "tab:green", linestyle=lines[i], label="Bias (d={:.0f})".format(deg[i]))

    plt.legend()

    if log==True: plt.semilogy()
    fig.savefig("output/figures/{:}_bias_variance_{:}_deg_{:}.pdf".format(RegType, x_type, info))
    fig.savefig("output/figures/{:}_bias_variance_{:}_deg_{:}.png".format(RegType, x_type, info))
    print("    Figure saved in: output/figures/{:}_bias_variance_{:}_deg_{:}.pdf\n".format(RegType,x_type, info))
    plt.close()

def all_metrics_test_train(x, metrics_test, metrics_train, x_type="", reg_type="", other="", info=""):
    """
    --------------------------------
    Input
        other: additional info for the title of the plot
    --------------------------------
    TODO: change back to saving in PDF format
    """

    print("Plotting all metrics for {:} ({:})".format(reg_type, info))
    n_plots = metrics_test.shape[0]     # Number of subplots

    titles = ["Explained R2-score", "Mean Squared Error", "Variance", "Bias"]
    y_labels = ["R2", "MSE", "Variance", "Bias"]
    x_label = "Model Complexity (Degrees)"
    if x_type == "data": x_label = "Size of dataset"

    fig, ax = plt.subplots(nrows=n_plots, ncols=1, sharex="col", sharey=False, figsize=[6.4, 6.4], constrained_layout=True) #[6.4, 4.8].

    fig.suptitle("{:} metrics {:}".format(reg_type, other), fontsize = 14, fontname = "serif")

    # R-score without log axis
    ax[0].grid()
    ax[0].plot(x, metrics_test[0], color="tab:green", label="Test")
    ax[0].plot(x, metrics_train[0], color="tab:blue", label="Train")
    ax[0].set_ylabel(y_labels[0],fontsize = 10, fontname = "serif" )
    if len(x) < 10: ax[0].set_xticks(x)

    for i in range(1, n_plots):
        ax[i].grid()
        ax[i].plot(x, metrics_test[i], color="tab:green", label="Test")
        ax[i].plot(x, metrics_train[i], color="tab:blue", label="Train")
        ax[i].semilogy()
        ax[i].set_ylabel(y_labels[i],fontsize = 10, fontname = "serif" )
        if len(x) < 10: ax[i].set_xticks(x)

    lines = []
    labels = []

    for ax in fig.axes:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)

    fig.legend(lines[:2], labels[:2], loc = 'upper right')
    plt.xlabel(x_label)

    fig.savefig("output/figures/{:}_metrics_test_train_{:}_{:}.pdf".format(reg_type, x_type, info))
    fig.savefig("output/figures/{:}_metrics_test_train_{:}_{:}.png".format(reg_type, x_type, info))
    print("    Figure saved in: output/figures/{:}_metrics_test_train_{:}_{:}.png\n".format(reg_type, x_type, info))
    plt.close()

def bias_variance(x, mse, var, bias, x_type="degrees", RegType ="OLS", info="", log=False):
    """
    Plots the Bias, Variance and MSE as a function of either
    degrees: x_type = "degrees",
    nummber of datapoints: x_type = "data"
    lambdas: x_type = "lambda"
    --------------------------------
    Input
        x, mse, var, bias, x_type="degrees", RegType ="OLS", info="", log=False
    --------------------------------
    TODO: change back to saving in PDF format
    """
    print("Plotting the Bias, Variance and MSE for {:} as a function of {:}".format(RegType, x_type))
    fig = plt.figure()
    plt.grid()
    plt.title("{:} Bias-Variance".format(RegType), fontsize = 14, fontname = "serif")

    if x_type == "degrees": plt.xlabel("Model complexity (degrees)", fontsize = 12, fontname = "serif")
    if x_type == "data": plt.xlabel("Number of datapoints (n)", fontsize = 12, fontname = "serif")
    if x_type == "lambda": plt.xlabel("$\lambda$ (log scale)", fontsize = 12, fontname = "serif")

    plt.plot(x, mse, "tab:red", label="MSE")
    plt.plot(x, var, "tab:blue", label="Variance")
    plt.plot(x, bias, "tab:green", label="Bias")
    plt.legend()

    if log==True: plt.semilogy()
    fig.savefig("output/figures/{:}_bias_variance_{:}_{:}.pdf".format(RegType, x_type, info))
    fig.savefig("output/figures/{:}_bias_variance_{:}_{:}.png".format(RegType, x_type, info))
    print("    Figure saved in: output/figures/{:}_bias_variance_{:}_{:}.pdf\n".format(RegType,x_type, info))
    plt.close()

def beta_conf(beta, conf_beta, d, mse_best, r2_best, rType="OLS", lamb = 0, info = ""):

    print("Plotting the {:} regression parameters with confidence intervals".format(rType))
    fig, ax = plt.subplots()
    ax.grid()

    plt.title("{:} parameters (d={:.0f}, MSE={:.3f}, R2={:.2f}) ".format(rType, d, mse_best, r2_best), fontsize = 14, fontname = "serif")
    if rType!= "OLS":
            plt.title("{:} parameters (d={:.0f}, MSE={:.3f}, R2={:.2f}, $\lambda$ = {:.3f}) ".format(rType, d, mse_best, r2_best, lamb), fontsize = 14, fontname = "serif")

    x = np.arange(len(beta))
    plt.errorbar(x, beta, yerr=conf_beta.ravel(), markersize=4, linewidth=1, capsize=5, capthick=1, ecolor="black", fmt='o')
    xlabels = [r"$\beta_"+"{:.0f}$".format(i) for i in range(len(beta))]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)

    if rType == "OLS":
        fig.savefig("output/figures/{:}_parameters_pdeg{:.0f}_{:}.pdf".format(rType, d, info))
        fig.savefig("output/figures/{:}_parameters_pdeg{:.0f}_{:}.png".format(rType, d, info))
        print("    Figure saved in: output/figures/{:}_parameters_pdeg{:.0f}_{:}.pdf\n".format(rType, d, info))
    else:
        fig.savefig("output/figures/{:}_parameters_pdeg{:.0f}_lamb{:.0f}_{:}.pdf".format(rType, d, lamb, info))
        fig.savefig("output/figures/{:}_parameters_pdeg{:.0f}_lamb{:.0f}_{:}.png".format(rType, d, lamb, info))
        print("    Figure saved in: output/figures/{:}_parameters_pdeg{:.0f}_lamb{:.0f}_{:}.pdf\n".format(rType, d, lamb, info))
    plt.close()

def RIDGE_beta_conf(beta, conf_beta, d, mse_best, r2_best, lamb="0", info=""):

    print("Plotting the Ridge regression parameters with confidence intervals")
    fig, ax = plt.subplots()
    ax.grid()

    plt.title("Ridge parameters (d={:.0f}, MSE={:.3f}, R2={:.2f}, $\lambda$=) ".format(d, mse_best, r2_best, lamb), fontsize = 12, fontname = "serif")
    x = np.arange(len(beta))
    plt.errorbar(x, beta, yerr=conf_beta.ravel(), markersize=4, linewidth=1, capsize=5, capthick=1, ecolor="black", fmt='o')
    xlabels = [r"$\beta_"+"{:.0f}$".format(i) for i in range(len(beta))]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)

    fig.savefig("output/figures/RIDGE_parameters_pdeg{:.0f}_lamb{:.0f}_{:}.pdf".format(d, lamb, info))
    fig.savefig("output/figures/RIDGE_parameters_pdeg{:.0f}_lamb{:.0f}_{:}.png".format(d, lamb, info))
    print("    Figure saved in: output/figures/RIDGE_parameters_pdeg{:.0f}_lamb{:.0f}_{:}.png".format(d, lamb, info))
    plt.close()


if __name__ == '__main__':
    plot_franke("Illustration of the Franke Function", "franke_func_illustration", 0.1)
    x = np.linspace(0, 10, 11)
    y1 = [0, 2*x, 2*x+1, 2*x+2]
    y2 = [0, 3*x, 3*x+1, 3*x+2]
    y3 = [0, 4*x, 4*x+1, 4*x+2]

    #z1 = 2*x
    #z2 = 3*x
    #z3 = 4*x
    bias_variance_m(x, y1, y2, y3,1,2,3, x_type="degrees", RegType ="OLS", info="test", log=False)
    compare_MSE(x, z1, z2, z3, rType = "OLS", info="test", log=True)
    #all_metrics_test_train(x, y, z, x_type="degrees", reg_type="OLS", other="Bootstrap", info="k", log=False)
