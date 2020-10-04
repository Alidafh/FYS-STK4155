#!/usr/bin/python
import numpy as np
import plotting as plot
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import scipy as scl
from tools import SVDinv
import sys
###############################################################################

def FrankeFunction(x,y):
    """
    Gives the values f(x,y) of the franke function
    --------------------------------
    Input
        x: numpy array or scalar
        y: numpy array or scalar
    --------------------------------
    TO DO: FINISHED
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
    return
        x: random numpy array [0,1]
        y: random numpy array [0,1]
        z: data from the franke function at given x and y values
    TO DO: FINISHED
    """

    if seed == "debug":
        np.random.seed(42)
        print("Running in debug mode")
        print("Generating data for the Franke function with n = {:.0f} datapoints\n".format(nData))

    x = np.random.rand(nData, 1)
    y = np.random.rand(nData, 1)

    z = FrankeFunction(x, y)

    if noise_str != 0:
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

    TO DO: FINISHED
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
    Calculate the R^2 score, mean square error, variance and bias.
    If the predicted values has shape (n,1), it tests the calculated MSE and R2
    values against results from Scikitlearn. If the predicted values have shape
    (n,m) it checks that the calculated bias+variance <= MSE. Nothing is printed
    if you pass the test.
    --------------------------------
    Input
        z_true: The true response value
        z_approx: The approximation found using regression
        test: If you want to test the calculations (default is False)
    --------------------------------
    return: R2, MSE, var, bias
    TODO: When using this with bootstrap, the calculated R2 score is waay off
          and when using it with kFold the bias and the MSE are identical
          #R2 = 1-((z_true-z_pred).T@(z_true-z_pred))/((z_true-np.mean(z_true)).T@(z_true-np.mean(z_true)))
    """

    #if np.shape(z_pred) != (len(z_true), 1)
    n = len(z_true)
    R2= 1 - ((np.sum((z_true - z_pred)**2))/(np.sum((z_true - np.mean(z_true))**2)))
    MSE = np.mean(np.mean((z_true - z_pred)**2, axis=1, keepdims=True))
    bias = np.mean((z_true - np.mean(z_pred, axis=1, keepdims=True))**2)
    var = np.mean(np.var(z_pred, axis=1, keepdims=True))
    if np.shape(z_pred) == (n, 1):
        var = np.mean(np.var(z_pred, axis=0, keepdims=True))

    if test == True:
        if np.shape(z_pred) == (len(z_pred), 1):
            #print("Testing with sklearn:")
            r2_sklearn = r2_score(z_true, z_pred)
            mse_sklearn = mean_squared_error(z_true, z_pred)

            if np.abs(R2-r2_sklearn) > 0.001:
                print("Testing with sklearn:")
                print("     Diff R2 : {:.2f}". format(R2-r2_sklearn))

            if np.abs(MSE-mse_sklearn) > 0.001:
                print("     Diff MSE: {:.2f}\n".format(MSE-mse_sklearn))
        else:
            if np.abs((bias+var) - MSE) > 1e-6:
                print("Test:")
                print("bias+variance > mse:")
                print("-------------")
                print("MSE: ", MSE)
                print("bias:", bias)
                print("var: ", var)
                print("-------------\n")

    return R2, MSE, var, bias
    #return r2_sklearn, mse_sklearn, var, bias

def OLS_SVD(z, X, var = False):
    """
    Preforming ordinary least squares fit to find the regression parameters
    using a signgular value decomposition. Also, if prompted it calculates the
    variance of the fitted parameters
    --------------------------------
    Input
        z: response variable
        X: Design matrix
        var: To calculate the variance set this to True (default is False)
    --------------------------------
    TODO: Finished
    """

    U, D, Vt = np.linalg.svd(X)
    V = Vt.T
    diagonal = np.zeros([V.shape[1], U.shape[1]])
    np.fill_diagonal(diagonal, D**(-1))
    beta = (V @ diagonal @ U.T) @ z

    if len(z) < len(beta):
        print("ERROR: n = {} < p = {}". format(len(z), len(beta)))
        #print("Remember that OLS is not well defined for p > n!")

    if var == True:
        diagonal_var = np.zeros([V.shape[0], V.shape[1]])
        np.fill_diagonal(diagonal_var, D**(-2))

        z_pred = X @ beta
        sigma2 = np.sum((z - z_pred)**2)/(len(z)-len(beta)-1)
        #sigma2 = np.sum((z - z_pred)**2)/(len(z)-len(beta))
        if sigma2 <= 0:
            sigma2 = np.abs(sigma2)

        var_beta = sigma2 * (np.diag(V @ diagonal_var @ Vt)[np.newaxis]).T
        return beta, var_beta.ravel()
    return beta

def OLS(z, X, var = False):
    """
    Preforming ordinary least squares fit to find the regression parameters
    beta. If prompted it calculates the variance of the fitted parameters.
    An error message will be printed if the design matrix has high dimentionality,
    p > n, but the parameters are still calculated. As this would give a negative
    variance, a temporary workaround is to take the absolute value of sigma^2.
    --------------------------------
    Input
        z: response variable
        X: Design matrix
        var: To calculate the variance set this to True (default is False)
    --------------------------------
    TODO: Find out if the absoultevalue thing in variance calculations is legit.
    """

    #beta = np.linalg.pinv(X) @ z
    beta = np.linalg.pinv(X.T @ X) @ X.T @ z

    if len(z) < len(beta):
        print("ERROR: n = {} < p = {}". format(len(z), len(beta)))
        print("Remember that OLS is not well defined for p > n!\n")

    if var == True:
        # Calculate the variance of the regression parameter
        z_pred = X @ beta

        sigma2 = np.sum((z - z_pred)**2)/(len(z)-len(beta)-1)
        #sigma2 = np.sum((z - z_pred)**2)/(len(z)-len(beta))
        if sigma2 <= 0:
            sigma2 = np.abs(sigma2)

        var_beta = sigma2 * SVDinv(X.T @ X).diagonal()
        return beta, var_beta
    return beta


def Ridge(z, X, lamb, var=False):
    """
    Preforming Pridge regression to find the regression parameters. If prompted
    it calculates the variance of the fitted parameters.
    --------------------------------
    Input
        z: response variable
        X: design matrix
        lamb: penalty parameter
        var: to calculate the variance set this to True (default is False)
    --------------------------------
    TODO: finish variance
    """
    I = np.eye(X.shape[1])
    beta = np.linalg.pinv( X.T @ X + lamb*I) @ X.T @ z

    if var==True:
        var_beta = 0
        return beta, var_beta

    return beta



if __name__ == '__main__':
    x, y, z = GenerateData(10, 0.01, "debug")
    X = PolyDesignMatrix(x, y, 2)

    # Test if the returned beta's are the same for small lambdas:
    beta_ridge, var_ridge = Ridge(z, X, 0.0000001, True)
    beta_ols, var_ols = OLS(z,X, True)
    print("Ridge: ", np.array_str(beta_ridge.ravel(), precision=2, suppress_small=True))
    print("OLS:   ", np.array_str(beta_ols.ravel(), precision=2, suppress_small=True))
