#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import PolynomialFeatures
###############################################################################

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def GenerateData(nData, start, stop, noise_str=0, seed=""):
    """
    Generates data for the Franke function with x,y:[start,stop]
    --------------------------------
    Input
        nData: number of datapoints
        start: minimum x,y
        stop: maximum x,y
        noiseStrength: the strength of the noise
        seed: if set to "debug" random numbers are the same for each turn
    --------------------------------
    TODO: Change back to saving plot in pdf format
    """
    if seed == "debug":
        np.random.seed(3155)
        print("Running in debug mode")

    steps = stop/nData
    x = np.arange(start, stop, steps)
    y = np.arange(start, stop, steps)
    x, y = np.meshgrid(x,y)

    print("Generating data with x,y:[{:},{:}] with {:} datapoints(step size {:.2e})".format(start, stop, nData, steps))
    data = FrankeFunction(x, y)
    if noise_str != 0:
        print("     Adding noise with {:} x normal distribution".format(noise_str))
        noise = noise_str*np.random.randn(len(x), 1)
        data += noise

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x,y,data,cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.title('Franke Function')
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$f(x,y)$")
    #plt.savefig("output/figures/franke_nData{}_noise{}.pdf".format(nData, noise_str), scale=0.1)
    plt.savefig("output/figures/franke_nData{}_noise{}.png".format(nData, noise_str), scale=0.1)
    print("     Figure saved in: output/figures/franke_nData{}_noise{}.pdf\n".format(nData, noise_str))
    #plt.show()
    return x, y, data

if __name__ == '__main__':
    x, y, data = GenerateData(5, 0, 1, 0.1, "debug")
