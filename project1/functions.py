#!/usr/bin/python
import numpy as np
import plotting as plot
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.preprocessing import StandardScaler
import scipy as scl
from tools import SVDinv
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
        print("Generating data for the Franke function with n = {:.0f} datapoints\n".format(nData))

    x = np.random.rand(nData, 1)
    y = np.random.rand(nData, 1)

    z = FrankeFunction(x, y)
    if noise_str != 0:
        #noise = noise_str * np.random.randn(nData, 1)
        noise = np.random.normal(0, noise_str, z.shape)
        z += noise

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
    """
    Scales the training and test data using sklearn StandardScaler
    --------------------------------
    Input
        train: The training set
        test: The test set
    --------------------------------
    returns train_scl and test_scl
    """
    scaler = StandardScaler()
    scaler.fit(train[:,1:])
    train_scl =  np.ones(train.shape)
    test_scl = np.ones(test.shape)
    train_scl[:,1:] = scaler.transform(train[:,1:])
    test_scl[:,1:] = scaler.transform(test[:,1:])
    return train_scl, test_scl

def metrics(z_true, z_pred, test=False):
    """
    Calculate the R^2 score, mean square error, and variance. The calculated
    R2-score and MSE are compared to the results from sklearn.
    --------------------------------
    Input
        z_true: The true response value
        z_approx: The approximation found using regression
        test: Test the calculations with the sklearn values (default is False)
    --------------------------------
    return: R2, MSE, var, bias
    TODO: quit if difference is too large?
    """
    n = len(z_true)
    # Calculate the r2-score, mean squared error, variance and bias
    R2 = 1 - ((np.sum((z_true - z_pred)**2))/(np.sum((z_true - np.mean(z_true, keepdims=True))**2)))
    MSE = np.mean(np.mean((z_true - z_pred)**2, axis=1, keepdims=True))
    var = np.mean(np.var(z_pred, axis=1, keepdims=True))
    bias = np.mean((z_true - np.mean(z_pred, axis=1, keepdims=True))**2)
    if test == True:
        r2_sklearn = r2_score(z_true.ravel(), z_pred.ravel())
        mse_sklearn = mean_squared_error(z_true.ravel(), z_pred.ravel())
        # Double check answers:
        if np.abs(R2-r2_sklearn) > 0.001:
            print("     Diff R2 : {:.2f}". format(R2-r2_sklearn))

        if np.abs(MSE-mse_sklearn) > 0.001:
            print("     Diff MSE: {:.2f}".format(MSE-mse_sklearn))

    return R2, MSE, var, bias

def OLS_SVD(z, X, var = False):
    """
    Preforming ordinary least squares fit to find the regression parameters
    using a signular value decomposition. Also, if prompted it calculates the
    variance of the fitted parameters
    --------------------------------
    Input
        z: response variable
        X: Design matrix
        var: True if you want to calculate the variance
    --------------------------------
    TODO: Make class called regression instead?
    """
    U, D, Vt = np.linalg.svd(X)
    V = Vt.T
    diagonal = np.zeros([V.shape[1], U.shape[1]])
    np.fill_diagonal(diagonal, D**(-1))
    beta = (V @ diagonal @ U.T) @ z     # Same as pinv

    if var == True:
        # Problem: if n datapoints is smaller than the number of parameters
        # the estimated sigma is negative and this code does not work.
        # Temporarely solved by setting sigma to zero. Will introduce bias tho??

        diagonal_var = np.zeros([V.shape[0], V.shape[1]])
        np.fill_diagonal(diagonal_var, D**(-2))

        z_pred = X @ beta
        sigma2 = np.sum((z - z_pred)**2)/(len(z)-len(beta)-1)
        if sigma2 <= 0:
            print("ERROR: n = {} < p = {}: n-p-1 = {}". format(len(z), len(beta), len(z)-len(beta)-1))
            sigma2 = np.abs(sigma2)

        var_beta = sigma2 * (np.diag(V @ diagonal_var @ Vt)[np.newaxis]).T
        return beta, var_beta.ravel()
    return beta

def OLS(z, X, var = False):
    """
    Preforming ordinary least squares fit to find the regression parameters
    beta. Uses the numpy pseudoinverse of X for inverting the matrix.
    If prompted it calculates the variance of the fitted parameters
    --------------------------------
    Input
        z: response variable
        X: Design matrix
    --------------------------------
    TODO: Make class called regression instead?
    """
    beta = np.linalg.pinv(X) @ z

    if var == True:
        # Problem: if n datapoints is smaller than the number of parameters
        # the estimated sigma is negative and this code does not work.
        # Temporarely solved by setting sigma to zero. Will introduce bias tho??
        z_pred = X @ beta
        sigma2 = np.sum((z - z_pred)**2)/(len(z)-len(beta)-1)
        if sigma2 <= 0:
            print("ERROR: n = {} < p = {}: n-p-1 = {}". format(len(z), len(beta), len(z)-len(beta)-1))
            sigma2 = np.abs(sigma2)

        var_beta = sigma2 * SVDinv(X.T @ X).diagonal()
        return beta, var_beta
    return beta


if __name__ == '__main__':
    x, y, z = GenerateData(10, 0.1, "debug")
    X = PolyDesignMatrix(x, y, 1)
    #beta = OLS(z, X)
    beta, conf = OLS(z, X, True)
    b2, ci2 = OLS_SVD(z,X, True)
    print(beta, conf)
    print("---------------")
    print(beta, ci2)
