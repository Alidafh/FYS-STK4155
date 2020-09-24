#!/usr/bin/python
import numpy as np
import plotting as plot

from sklearn.metrics import r2_score

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
    TODO: Change back to saving plot in pdf format, unmute print statements?
    """
    if seed == "debug":
        np.random.seed(3155)
        print("Running in debug mode")

    steps = stop/nData
    x = np.arange(start, stop, steps)
    y = np.arange(start, stop, steps)
    x, y = np.meshgrid(x,y)

    print("Generating data for the Franke function with n = {:.0f} datapoints".format(len(x)**2))
    z = FrankeFunction(x, y)
    if noise_str != 0:
        noise = noise_str*np.random.randn(len(x), 1)
        z += noise

    plot.plot_3D(x,y,z, "Franke Function", "franke_nData{}_noise{}".format(nData, noise_str))
    return x, y, z

def PolyDesignMatrix(x,y, degree):
    """
    Generates a design matrix of size (n,p) using a polynomial of chosen degree
    --------------------------------
    Input
        x: numpy array with shape (n,n)
        y: numpy array with shape (n,n)
        degree: the degree of the polynomial
    --------------------------------
    TODO: Cleanup and comment
    """
    if len(x.shape) > 1:    # Easier to use arrays with shape (n, 1) where n = m**2
                x = x.ravel()
                y = y.ravel()

    m = len(x)
    p = int(((degree+2)*(degree+1))/2)  # number of terms in beta
    X = np.ones((m, p))
    print("Generating {:.0f}nd degree polynomial design matrix of size (n, p) =".format(degree), np.shape(X))

    for i in range(1, degree+1):
        j = int(((i)*(i+1))/2)
        for k in range(i+1):
            # fill colums of X with the polynomials [1  x  y  x**2  y**2  xy ...]
            X[:,j+k] = x**(i-k)*y**(k)
    return X

def metrics(z_true, z_approx):
    """
    Calculate the R^2 score, mean square error, variance and bias
    --------------------------------
    Input
        z_true: The true response value
        z_approx: The approximation found using regression
    --------------------------------
    TODO: Write the code
    """
    if len(z_true.shape) > 1:   # Fix shape of the z_true array
        z_true = z_true.ravel()

    print("Calculating R2-score, mean squared error, variance and bias")
    R2 = 1 - ((np.sum((z_true-z_approx)**2))/(np.sum((z_true - np.mean(z_true))**2)))
    MSE = (1.0/(np.size(z_true))) *np.sum((z_true - z_approx)**2)
    var = 0
    bias = 0
    return R2, MSE, var, bias


def OLS(z, X):
    """
    Preforming ordinary least squares fit
    --------------------------------
    Input
        z: response variable
        X: Design matrix
    --------------------------------
    TODO: Make class called regression instead?
    """
    if len(z.shape) > 1:    # If the
        z = z.ravel()
    print("Prefomring OLS regression")
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z)
    z_approx = X @ beta
    return beta, z_approx


if __name__ == '__main__':
    x, y, z = GenerateData(20, 0, 1, 0.1, "debug")
    X = PolyDesignMatrix(x,y,2)
    print(np.shape(X))
    beta, z_approx = OLS(z, X)
    R2, MSE, var, bias = metrics(z, z_approx)
    print(R2, MSE, var, bias)
