#!/usr/bin/python
import numpy as np
import plotting as plot
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
#sklearn.metrics.explained_variance_score(y_true, y_pred,
#sklearn.metrics.mean_squared_error(y_true, y_pred,
from numba import jit
import scipy as scl
from sklearn.preprocessing import StandardScaler
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

def GenerateData(nData, noise_str=0, seed=""):
    """
    Generates data for the Franke function with x,y:[start,stop]
    --------------------------------
    Input
        nData: number of datapoints
        noise_str: the strength of the noise, default is zero
        seed: if set to "debug" random numbers are the same for each turn
    --------------------------------
    TODO: Change back to saving plot in pdf format, unmute print statements?
    """
    if seed == "debug":
        np.random.seed(42)
        print("Running in debug mode")

    x = np.random.rand(int(np.sqrt(nData)), 1)
    y = np.random.rand(int(np.sqrt(nData)), 1)
    x,y = np.meshgrid(x,y)

    print("Generating data for the Franke function with n = {:.0f} datapoints".format(nData))
    z = FrankeFunction(x, y)
    if noise_str != 0:
        noise = noise_str * np.random.randn(int(np.sqrt(nData)), 1)
        z += noise

    x = x.ravel()
    y = y.ravel()
    z = z.ravel()
    return x, y, z


def PolyDesignMatrix(x, y, degree):
    """
    Generates a design matrix of size (n,p) using a polynomial of chosen degree
    --------------------------------
    Input
        x: numpy array with shape (n,n)
        y: numpy array with shape (n,n)
        degree: the degree of the polynomial
    --------------------------------
    Returns
        X: Design matrix
    TODO: Cleanup and comment
    """
    if len(x.shape) > 1:    # Easier to use arrays with shape (n, 1) where n = m**2
        x = x.ravel()
        y = y.ravel()

    n = len(x)
    p = int(((degree+2)*(degree+1))/2)  # number of terms in beta
    X = np.ones((n, p))     # Matrix of size R^nxp where all entries are zero

    # fill colums of X with the polynomials [1  x  y  x**2  y**2  xy ...]
    for i in range(1, degree+1):
        j = int(((i)*(i+1))/2)
        for k in range(i+1):
            X[:,j+k] = x**(i-k)*y**(k)
    return X

def scale_X(train, test):
    scaler = StandardScaler()
    scaler.fit(train[:,1:])
    train_scl =  np.ones(train.shape)
    test_scl = np.ones(test.shape)
    train_scl[:,1:] = scaler.transform(train[:,1:])
    test_scl[:,1:] = scaler.transform(test[:,1:])
    return train_scl, test_scl

def metrics(z_true, z_pred, test=""):
    """
    Calculate the R^2 score, mean square error, and variance
    --------------------------------
    Input
        z_true: The true response value
        z_approx: The approximation found using regression
    --------------------------------
    return: R2, MSE, var, bias
    TODO:
    """
    n = z_true.size
    # Calculate the r2-score, mean squared error, variance and bias
    R2 = 1 - ((np.sum((z_true - z_pred)**2))/(np.sum((z_true - np.mean(z_true))**2)))
    MSE = np.sum((z_true - z_pred)**2)/n
    var = np.mean(np.var(z_pred, keepdims=True))
    bias = np.mean((z_true - np.mean(z_pred, keepdims=True))**2)

    if test == "test":
        r2_sklearn = r2_score(z_true, z_pred)
        mse_sklearn = mean_squared_error(z_true, z_pred)

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
    #beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z)
    beta = np.linalg.pinv(X) @ z

    #z_pred= X @ beta

    # Find 95% contidence intervals for the beta values
    #sigma2 = np.sum((z - z_pred)**2)/(len(z)-len(beta)-1)
    #var_beta = sigma2 * np.linalg.inv(X.T.dot(X)).diagonal()
    #std_beta = np.sqrt(var_beta)

    conf_inter = np.zeros((len(beta), 2))       # Matrix of zeros with dimention (p,2)
    #for i in range(len(beta)):
        #std_beta = np.sqrt(var_beta[i,i])
    #    conf_inter[i, 0] = np.mean(beta) - 2*std_beta[i]
    #    conf_inter[i, 1] = np.mean(beta) + 2*std_beta[i]

    return beta, conf_inter

if __name__ == '__main__':
    x, y, z = GenerateData(20, 0, 1, 0.1, "debug")
    X = PolyDesignMatrix(x,y,2)
    x_train, x_test, z_train, z_test = train_test_split(X, z, test_size=0.33)
    x_train_s = x_train - np.mean(x_train)
    x_test_s = x_test - np.mean(x_test)

    scaler = StandardScaler()
    scaler.fit(X_train[:,1:])
    X_train_scaled = scaler.transform(X_train[:,1:])
    #X_test_scaled = scaler.transform(X_test)

    n = len(X_train_scaled[:,1])
    ones = np.ones((n,1))

    X_train_new = np.hstack((ones, X_train_scaled))

    print(x_train_s)
