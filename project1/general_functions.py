#!/usr/bin/python
import numpy as np
import plotting as plot
###############################################################################

def FrankeFunction(x,y):
    """
    Gives the values f(x,y) of the franke function
    --------------------------------
    Input
        x: numpy array or scalar
        y: numpy array or scalar
    --------------------------------
    """
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
        noise_str: the strength of the noise, default is zero
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

    print("Generating data for the Franke function with m = {:.0f} datapoints".format(len(x)**2))
    z = FrankeFunction(x, y)
    if noise_str != 0:
        print("     Adding noise with {:} x normal distribution".format(noise_str))
        noise = noise_str*np.random.randn(len(x), 1)
        z += noise

    plot.plot_3D(x,y,z, "Franke Function", "franke_nData{}_noise{}".format(nData, noise_str))
    return x, y, z

def PolyDesignMatrix(x,y, degree):
    """
    Generates a design matrix of size (m,p) using a polynomial of chosen degree
    --------------------------------
    Input
        x: numpy array with shape (n,n)
        y: numpy array with shape (n,n)
        degree: the degree of the polynomial
    --------------------------------
    TODO: Cleanup and comment
    """

    x = x.ravel()   # Easier to use arrays with shape (m, 1) where m = n**2
    y = y.ravel()

    m = len(x)
    p = int(((degree+2)*(degree+1))/2)  # number of terms in beta
    X = np.ones((m, p))
    print("Generating polynomial design matrix of size (m, p) =", np.shape(X))

    # fill colums of X with the polynomials [1  x  y  x**2  y**2  xy ...]
    for i in range(1, degree+1):
        j = int(((i)*(i+1))/2)
        for k in range(i+1):
            X[:,j+k] = x**(i-k)*y**(k)

    return X

if __name__ == '__main__':
    x, y, z = GenerateData(2, 0, 1, 0.1, "debug")
    X = PolyDesignMatrix(x,y,2)
