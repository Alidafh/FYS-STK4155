#!/usr/bin/python
import numpy as np
import plotting as plot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import scipy as scl
from tools import SVDinv, foldIndex
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
    Returns:
        z: numpy array or scalar representing the function value
    --------------------------------
    TO DO: FINISHED
    """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    z = term1 + term2 + term3 + term4
    return z

def GenerateData(nData, noise_str=0, seed="", pr=True):
    """
    Generates three numpy arrays x, y, z of size (nData, 1).
    The x and y arrays are randomly distributed numbers between 0 and 1.
    The z array is created using the Franke Function with x and y, and if a
    noise_str is specified, random normally distributed noise with strength
    noise_str is added to the z-array.
    --------------------------------
    Input
        nData: number of datapoints
        noise_str: the strength of the noise, default is zero
        seed: if set to "debug" random numbers are the same for each turn
    --------------------------------
    Returns
        x: numpy array of shape (n,1) with random numbers between 0 and 1
        y: numpy arary of shape (n,1) with random numbers between 0 and 1
        z: numpy array of shape (n,1) with Franke function values f(x,y)
    --------------------------------
    TO DO: FINISHED
    """
    if pr==True:
        print("Generating data for the Franke function with n = {:.0f} datapoints\n".format(nData))

    np.random.seed(42)
    x = np.random.rand(nData, 1)
    y = np.random.rand(nData, 1)

    z = FrankeFunction(x, y)

    if noise_str != 0:
        noise = np.random.normal(0, noise_str, z.shape)
        z += noise

    return x, y, z

def PolyDesignMatrix(x, y, d):
    """
    Generates a design matrix of size (n,p) with monomials up to degree d as
    the columns. As an example if d=2 the columns of the design matrixs will be
    [1  x  y  x**2  y**2  xy].
    --------------------------------
    Input
        x: numpy array with shape (n,) or (n,1)
        y: numpy array with shape (n,) or (n,1)
        d: the degree of the polynomial (scalar)
    --------------------------------
    Returns
        X: Design matrix of shape (n, p)
    --------------------------------
    TO DO: FINISHED
    """
    if len(x.shape) > 1:
        x = x.ravel()
        y = y.ravel()

    n = len(x)
    p = int(((d+2)*(d+1))/2)  # number of terms in beta
    X = np.ones((n, p))       # Matrix of size (n,p) where all entries are zero

    # fill colums of X with the polynomials [1  x  y  x**2  y**2  xy ...]
    for i in range(1, d+1):
        j = int(((i)*(i+1))/2)
        for k in range(i+1):
            X[:,j+k] = x**(i-k)*y**(k)
    return X

def scale_X(train, test):
    """
    Scales the training and test data using sklearn's StandardScaler.
    --------------------------------
    Input
        train: The training set
        test:  The test set
    --------------------------------
    Returns
        train_scl: The scaled training set
        test_scl:  The scaled test set
    --------------------------------
    TO DO: FINISHED
    """
    train_scl =  np.ones(train.shape)
    test_scl = np.ones(test.shape)
    train_scl[:,1:] = train[:,1:] - np.mean(train)
    test_scl[:,1:]= test[:,1:] - np.mean(test)

    """
    scaler = StandardScaler()
    scaler.fit(train[:,1:])
    train_scl =  np.ones(train.shape)
    test_scl = np.ones(test.shape)
    train_scl[:,1:] = scaler.transform(train[:,1:])
    test_scl[:,1:] = scaler.transform(test[:,1:])
    """
    return train_scl, test_scl

def metrics(z_true, z_pred, test=False, quiet=True):
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
    Returns
        R2, MSE, var, bias
    --------------------------------
    TODO: When using this with bootstrap, the calculated R2 score is waay off
          and when using it with kFold the bias and the MSE are identical
    """
    n = len(z_true)
    if z_pred.shape != z_true.shape:
        print("metrics(): RESHAPING z_pred")
        z_pred = np.mean(z_pred, axis=1, keepdims=True)

    R2 = 1 - (np.sum((z_true - z_pred)**2))/(np.sum((z_true - np.mean(z_true))**2))
    MSE = np.mean(np.mean((z_true - z_pred)**2, axis=1, keepdims=True))
    bias = np.mean((z_true - np.mean(z_pred, axis=1, keepdims=True))**2)
    var = np.mean(np.var(z_pred, axis=1, keepdims=True))

    #if R2<0:
    #    print("metrics(): R2 is NEGATIVE: ", R2)

    if test == True:
        r2_sklearn = r2_score(z_true, z_pred)
        mse_sklearn = mean_squared_error(z_true, z_pred)

        if np.abs(R2-r2_sklearn) > 0.001:
            print("Testing with sklearn:")
            print("     Diff R2 : {:.2f}". format(R2-r2_sklearn))

        if np.abs(MSE-mse_sklearn) > 0.001:
            print("     Diff MSE: {:.2f}\n".format(MSE-mse_sklearn))

        #if np.abs((bias+var) - MSE) > 1e-6:
        if bias+var > MSE:
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
        z: response variable of shape (n,1) or (n,)
        X: Design matrix of shape (n,p)
        var: Bool. Set this to True to calculate the variance (default is False)
    --------------------------------
    Returns
        beta: The estimated OLS regression parameters shape (p,1)
        (var_beta: The variance of the parameters are returned if var=True)
    --------------------------------
    TO DO: FINISHED
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
    Preforming ordinary least squares fit to find the regression parameters.
    If prompted it also calculates the variance of the fitted parameters.
    An error message will be printed if the design matrix has high
    dimentionality, p > n, but the parameters are still calculated.
    As this would give a negative variance, a temporary workaround is to take
    the absolute value of sigma^2.
    --------------------------------
    Input
        z: response variable
        X: Design matrix
        var: To calculate the variance set this to True (default is False)
    --------------------------------
    Returns
    - var=False
        beta: The estimated OLS regression parameters, shape (p,1)
        (var_beta: The variance of the parameters (p,1), returned if var=True)
    --------------------------------
    TODO: Find out if the absoultevalue thing in variance calculations is legit.
    """

    #beta = np.linalg.pinv(X) @ z
    beta = np.linalg.pinv(X.T @ X) @ X.T @ z # To be consistent with Ridge

    if len(z) < len(beta):
        print("ERROR: n = {} < p = {}". format(len(z), len(beta)))
        print("Remember that OLS is not well defined for p > n!\n")

    if var == True:
        z_pred = X @ beta
        sigma2 = np.sum((z - z_pred)**2)/(len(z)-len(beta)-1)
        #sigma2 = np.sum((z - z_pred)**2)/(len(z)-len(beta))

        if sigma2 <= 0: sigma2 = np.abs(sigma2)

        var_beta = sigma2 * SVDinv(X.T @ X).diagonal()

        return beta, var_beta.reshape(len(var_beta),1)

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
    Returns
        beta: The estimated Ridge regression parameters with shape (p,1)
        (var_beta: The variance of the parameters (p,1), returned if var=True)
    --------------------------------
    TODO: check if it should be 1/(n-p) instead of 1/(n-p-1)
    """

    I = np.eye(X.shape[1])  # Identity matrix - (p,p)
    beta = np.linalg.pinv( X.T @ X + lamb*I) @ X.T @ z

    if var==True:
        z_pred = X @ beta
        sigma2 = np.sum((z - z_pred)**2)/(len(z)-len(beta)-1)
        #sigma2 = np.sum((z - z_pred)**2)/(len(z)-len(beta))

        a = np.linalg.pinv( X.T @ X + lamb*I)
        var_beta = sigma2 * (a @ (X.T @ X) @ a.T).diagonal()
        return beta, var_beta.reshape(len(var_beta), 1)

    return beta

def Bootstrap_v1(x, y, z, d, n_bootstraps, RegType, lamb=0):
    """
    THIS DOES NOT WORK
    --------------------------------

    --------------------------------
    Returns
        z_train, z_test, z_fit, z_pred
    --------------------------------
    TODO:
    """
    X = PolyDesignMatrix(x, y, d)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.33)
    #X_train, X_test = scale_X(X_train, X_test)

    z_pred = np.empty((z_test.shape[0], n_bootstraps))
    z_fit = np.empty((z_train.shape[0], n_bootstraps))

    for j in range(n_bootstraps):
        """ Loop over bootstraps"""
        tmp_X_train, tmp_z_train = resample(X_train, z_train)
        #X_train_scl, X_test_scl = scale_X(tmp_X_train, tmp_X_test)
        if RegType == "OLS": tmp_beta = OLS(tmp_z_train, tmp_X_train)
        if RegType == "RIDGE": tmp_beta = Ridge(tmp_z_train, tmp_X_train, lamb)
        z_pred[:,j] = X_test @ tmp_beta.ravel()
        z_fit[:,j] = tmp_X_train @ tmp_beta.ravel()

    return z_train, z_test, z_fit, z_pred

def Bootstrap(x, y, z, d, n_bootstraps, RegType="OLS", lamb=0):
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.33, shuffle=True)
    z_test_cp = np.zeros((z_test.shape[0], n_bootstraps))
    z_train_cp = np.zeros((z_train.shape[0], n_bootstraps))

    for i in range(n_bootstraps):
        z_test_cp[:,i] = z_test.ravel()

    z_pred = np.empty((z_test.shape[0], n_bootstraps))
    z_fit = np.empty((z_train.shape[0], n_bootstraps))

    for i in range(n_bootstraps):
        x_, y_,z_ = resample(x_train, y_train, z_train)
        z_train_cp[:,i] = z_.ravel()
        X_train = PolyDesignMatrix(x_, y_, d)
        X_test = PolyDesignMatrix(x_test, y_test, d)
        X_test, X_train = scale_X(X_test, )
        if RegType == "OLS" : tmp_beta = OLS(z_, X_train)
        if RegType == "RIDGE": tmp_beta = Ridge(z_, X_train, lamb)

        z_pred[:,i] = X_test @ tmp_beta.ravel()
        z_fit[:,i] = X_train @ tmp_beta.ravel()

    return z_train, z_test, z_fit, z_pred

def OLS_bootstrap_degrees(x, y, z, degrees, n_bootstraps, dim=0):
    """
    If you don't want to loop over many degrees set up the degrees array as
            degrees = np.arange(d, d+1)
    say if you want to only do it for d=2:
            degrees = np.arange(2, 2+1)
    --------------------------------
    Input
    --------------------------------
    returns:
        metrics_test:  (4, len(degrees))
        metrics_train: (4, len(degrees))
    --------------------------------
    TODO: describe
    """
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.33, shuffle=True)
    z_test_cp = np.zeros((z_test.shape[0], n_bootstraps))
    z_train_cp = np.zeros((z_train.shape[0], n_bootstraps))

    d_max = len(degrees)

    metrics_test = np.zeros((4, d_max))
    metrics_train = np.zeros((4, d_max))

    for i in range(n_bootstraps):
        z_test_cp[:,i] = z_test.ravel()

    z_pred = np.empty((z_test.shape[0], n_bootstraps))
    z_fit = np.empty((z_train.shape[0], n_bootstraps))

    for d in range(d_max):
        for i in range(n_bootstraps):
            x_, y_, z_ = resample(x_train, y_train, z_train)
            z_train_cp[:,i] = z_.ravel()
            X_train = PolyDesignMatrix(x_, y_, degrees[d])
            X_test = PolyDesignMatrix(x_test, y_test, degrees[d])
            tmp_beta = OLS(z_, X_train)
            z_pred[:,i] = X_test @ tmp_beta.ravel()
            z_fit[:,i] = X_train @ tmp_beta.ravel()

        metrics_test[:,d] = metrics(z_test_cp, z_pred, test=True)
        metrics_train[:,d] = metrics(z_train_cp, z_fit, test=True)

    return metrics_test, metrics_train

def OLS_bootstrap_data(ndata, n_bootstraps, d_1):

    degree_1 = np.arange(d_1, d_1+1)

    m_test_ndata = np.zeros((4, len(ndata)))
    m_train_ndata = np.zeros((4, len(ndata)))

    for i in range(len(ndata)):
        x_tmp, y_tmp, z_tmp = GenerateData(ndata[i], 0.01, "debug", pr=False)

        # Bootstrapping
        m_test_tmp, m_train_tmp = OLS_bootstrap_degrees(x_tmp, y_tmp, z_tmp, degree_1, n_bootstraps)

        m_test_ndata[:,i] = m_test_tmp.ravel()
        m_train_ndata[:,i] = m_train_tmp.ravel()

    return m_test_ndata, m_train_ndata

def kFold1(x, y, z, d=3, k=5, RegType="OLS", lamb=0):
    """
    """
    np.random.seed(42)
    np.random.shuffle(x)
    np.random.shuffle(y)
    np.random.shuffle(z)

    X = PolyDesignMatrix(x, y, d)
    dummy = np.array(([1,4],[1,4]))
    X, dummy = scale_X(X, dummy)

    #np.random.seed(42)
    #np.random.shuffle(X)

    z_pred, z_fit, z_test_cp, z_train_cp = [],[],[],[]
    for i in range(1, k+1):
        train_index, test_index = foldIndex(z, i, k)
        X_train = X[train_index]
        X_test = X[test_index]

        z_train = z[train_index]
        z_test = z[test_index]

        z_test_cp.append(z_test)
        z_train_cp.append(z_train)

        if RegType == "OLS": beta = OLS(z_train, X_train)
        if RegType == "RIDGE": beta = Ridge(z_train, X_train, lamb)

        z_fit_tmp = X_train @ beta
        z_pred_tmp = X_test @ beta

        z_pred.append(z_pred_tmp)
        z_fit.append(z_fit_tmp)

    z_test_k = np.zeros((len(z_test_cp[0]), k))
    z_train_k = np.zeros((len(z_train_cp[0]), k))
    z_pred_k = np.zeros((len(z_pred[0]), k))
    z_fit_k = np.zeros((len(z_fit[0]), k))

    for i in range(k):
        z_test_k[:,i] = z_test_cp[i].ravel()
        z_train_k[:,i] = z_train_cp[i].ravel()
        z_pred_k[:,i] = z_pred[i].ravel()
        z_fit_k[:,i] = z_fit[i].ravel()

    metrics_test =  metrics(z_test_k, z_pred_k, test=True)
    metrics_train = metrics(z_train_k, z_fit_k, test=True)

    return metrics_test, metrics_train


def kFold(x,y,z, d=3, k=5, RegType="OLS", lamb=0):

    z_pred = np.zeros((20,k))
    z_fit = np.zeros((80,k))

    z_test = np.zeros((20,k))
    z_train = np.zeros((80,k))

    np.random.seed(42)
    np.random.shuffle(x)
    np.random.shuffle(y)
    np.random.shuffle(z)

    j = 0
    for i in range(1, k+1):
        train_i, test_i = foldIndex(x, i, k)
        x_test, y_test , z_test_tmp = x[test_i], y[test_i], z[test_i]
        x_train, y_train, z_train_tmp = x[train_i], y[train_i], z[train_i]

        X_train = PolyDesignMatrix(x_train, y_train, d)
        X_test = PolyDesignMatrix(x_test, y_test, d)

        X_train, X_test = scale_X(X_train, X_test)

        if RegType == "OLS" : tmp_beta = OLS(z_train_tmp, X_train)
        if RegType == "RIDGE": tmp_beta = Ridge(z_train_tmp, X_train, lamb)

        z_pred[:,j] = X_test @ tmp_beta.ravel()
        z_fit[:,j] = X_train @ tmp_beta.ravel()

        z_test[:,j] = z_test_tmp.ravel()
        z_train[:,j] = z_train_tmp.ravel()

        j += 1

    return z_train, z_test, z_fit, z_pred

def kFold_degrees2(x, y, z, degrees, k=5, RegType="OLS", lamb=0):
    """
    Uses folding to split the data
    --------------------------------
    Input
        x,y,z : data
        degrees = np.array with degrees
    --------------------------------
    TO DO:
    """
    d = len(degrees)

    m_test = np.zeros((4, d))
    m_train = np.zeros((4,d))

    for i in range(d):
        """loop over degrees"""
        degree = degrees[i]
        z_train, z_test, z_fit, z_pred = kFold(x,y,z, degrees[i], k, RegType, lamb)
        m_test[:,i] = metrics(z_test, z_pred, test=True)
        m_train[:,i] = metrics(z_train, z_fit, test=True)

        #m_test[:,i], m_train[:,i] = kFold1(x, y, z, degrees[i], k, RegType, lamb)

    return m_test, m_train


if __name__ == '__main__':
    x, y, z = GenerateData(100, 0.1, "debug")
    degrees = np.arange(1, 11)
    #m_test = kFold_degrees(x, y, z, degrees, k=5, shuffle = False, RegType="OLS", lamb=0)
    #m_test, m_train = kFold(x, y, z, d=10, k=5)
    #print(m_test)
    #print(m_train)

    m_test, m_train = kFold_degrees2(x, y, z, degrees, k=5, RegType="OLS", lamb=0)
    #plt.plot(degrees, m_test[0])
    #plt.plot(degrees, m_train[0])

    plt.figure()
    plt.plot(degrees, m_test[1], label="MSE test")
    plt.plot(degrees, m_train[1], label="MSE train")
    plt.plot(degrees, m_test[2], label="variance")
    plt.plot(degrees, m_test[3], label="bias")
    plt.legend()
    plt.semilogy()
    plt.show()

    plt.figure()
    plt.plot(degrees, m_train[0])
    plt.plot(degrees, m_test[0])
    plt.show()

    """

    n_bootstraps = 100
    d_max = 2
    degrees = np.arange(2, d_max+1)
    print(degrees)
    m_test, m_train = OLS_bootstrap_degrees(x, y, z, degrees, n_bootstraps)
    print(m_test.shape, m_train.shape)
    print("--------------")
    print(m_test)

    plt.plot(degrees, m_test[1], label="MSE test")
    plt.plot(degrees, m_train[1], label="MSE train")
    plt.plot(degrees, m_test[2], label="variance")
    plt.plot(degrees, m_test[3], label="bias")
    plt.legend()
    plt.semilogy()
    plt.show()
    """

    """
    X = PolyDesignMatrix(x, y, 2)

    # Test if the returned beta's are the same for small lambdas:
    beta_ridge, var_ridge = Ridge(z, X, 0.0000000001, True)
    beta_ols, var_ols = OLS(z,X, True)

    print("Ridge: ", np.array_str(beta_ridge.ravel(), precision=2, suppress_small=True))
    print("OLS:   ", np.array_str(beta_ols.ravel(), precision=2, suppress_small=True))

    print("Ridge: ", np.array_str(var_ridge.ravel(), precision=2, suppress_small=True))
    print("OLS:   ", np.array_str(var_ols.ravel(), precision=2, suppress_small=True))
    """
