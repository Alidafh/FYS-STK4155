#!/usr/bin/python
import matplotlib.pyplot as plt
from matplotlib import cm
import general_functions as func
import numpy as np

def plot_franke(title, filename):
    """
    makes a 3d plot and saves it in the folder: output/figures/
    --------------------------------
    Input
        x,y,z:
        title: title of the figure
        filename: the name of the saved file
    --------------------------------
    TODO: change back to saving in PDF format
    """
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x,y)

    z = func.FrankeFunction(x,y)

    fig = plt.figure()
    #plt.style.use("classic")
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x,y,z,cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.title(title, fontsize = 12, fontname = "serif")
    ax.set_xlabel(r"$x$", fontsize = 12, fontname = "serif")
    ax.set_ylabel(r"$y$", fontsize = 12, fontname = "serif")
    ax.set_zlabel(r"$z$", fontsize = 12, fontname = "serif")
    #plt.savefig("output/figures/{}.pdf".format(filename), scale=0.1)
    plt.savefig("output/figures/{}.png".format(filename), scale=0.1)
    print("Figure saved in: output/figures/{}.pdf\n".format(filename))
    #plt.show()

def plot_MSE(x, test_error, train_error, type=""):
    fig = plt.figure()
    #plt.style.use("seaborn")
    plt.grid()
    plt.title("{} Mean Squared Errors".format(type), fontsize = 12, fontname = "serif")
    plt.plot(x, test_error, "tab:green", label="Test Error")
    plt.plot(x, train_error,"tab:blue", label="Train Error")
    plt.legend()
    plt.xlabel("Model complexity (degrees)", fontsize = 12, fontname = "serif")
    plt.ylabel("MSE", fontsize = 12, fontname = "serif")
    plt.savefig("output/figures/MSE_{}.png".format(type))
    print("Figure saved in: output/figures/MSE_{}.png\n".format(type))
    #plt.show()


if __name__ == '__main__':
    plot_franke("Illustration of the Franke Function", "franke_func_illustration")
