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

def GenerateData(nData, noise_str=0, pr=True):
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

def scale_X(train, test, scaler="manual"):
    """
    Scales the training and test data using sklearn's StandardScaler.
    --------------------------------
    Input
        train: The training set
        test:  The test set
        scaler: Choose what scaler you want to use
    --------------------------------
    Returns
        train_scl: The scaled training set
        test_scl:  The scaled test set
    --------------------------------
    TO DO: FINISHED
    """
    """
    if scaler == "sklearn":
        scaler = StandardScaler()
        scaler.fit(train[:,1:])
        train_scl =  np.ones(train.shape)
        test_scl = np.ones(test.shape)
        train_scl[:,1:] = scaler.transform(train[:,1:])
        test_scl[:,1:] = scaler.transform(test[:,1:])
    """

    train_scl =  np.ones(train.shape)
    test_scl = np.ones(test.shape)
    mean_train = np.mean(train[:,1:])
    std_train = np.std(train[:,1:])
    train_scl[:,1:] = (train[:,1:] - mean_train)/std_train
    test_scl[:,1:]= (test[:,1:] - mean_train)/std_train

    """
    train_scl =  np.ones(train.shape)
    test_scl = np.ones(test.shape)
    mean_train = np.mean(train[:,1:])
    #mean_train = np.mean(train)
    train_scl[:,1:] = (train[:,1:] - mean_train)
    test_scl[:,1:]= (test[:,1:] - mean_train)

    """
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

        if np.abs(R2-r2_sklearn) > 0.1:
            print("Testing with sklearn:")
            print("     R2_sklearn =", r2_sklearn)
            print("     R2_manual =", R2)
            print("     Diff R2 = {:.2f}". format(R2-r2_sklearn))
            print("-------------")
            print("    MSE = ", MSE)
            print("    bias+variance = ", bias+var)
            print("-------------\n")

        if np.abs(MSE-mse_sklearn) > 0.1:
            print("     Diff MSE = {:.2f}\n".format(MSE-mse_sklearn))
            print("-------------")
            print("    MSE = ", MSE)
            print("    bias+variance = ", bias+var)
            print("-------------\n")

    return R2, MSE, var, bias

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

def lasso(dm, z, lam):
    reg = Lasso(alpha=lam, fit_intercept = False, max_iter=100000)
    reg.fit(dm,z)
    # print(reg.intercept_)
    return np.transpose(np.array([reg.coef_]))

def OLSskl(dm, z, dummy):
    reg = LinearRegression()
    reg.fit(dm, z)
    # print(reg.intercept_)
    return np.transpose(np.array([reg.coef_]))

def Bootstrap(x, y, z, d, n_bootstraps, RegType, lamb=0):
    """
    Bootstrap loop v1 with design matrix outide loop
    --------------------------------
    Input
        x,y,z:        Variables for data from Generate Data
        d:            Polynomial degree for the feature matrix
        n_bootstraps: The number of bootstraps
        RegType:      "OLS" for Ordinary least squares(default)
                      "RIDGE" for Ridge
        lamb:         the lambda value if RegType="RIDGE"
    --------------------------------
    Returns
        z_train_cp: (train_size, n_bootstraps)
        z_test_cp:  (test_size, n_bootstraps)
        z_fit:      (train_size, n_bootstraps)
        z_pred      (test_size, n_bootstraps)
    --------------------------------
    TODO: Find out why it doesnt work for degrees 7+
    """
    X = PolyDesignMatrix(x, y, d)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.33)
    X_train, X_test = scale_X(X_train, X_test)

    z_pred = np.empty((z_test.shape[0], n_bootstraps))
    z_fit = np.empty((z_train.shape[0], n_bootstraps))

    z_test_cp = np.zeros((z_test.shape[0], n_bootstraps))
    z_train_cp = np.zeros((z_train.shape[0], n_bootstraps))

    for i in range(n_bootstraps):
        z_test_cp[:,i] = z_test.ravel()

    for j in range(n_bootstraps):
        tmp_X_train, tmp_z_train = resample(X_train, z_train)
        z_train_cp[:,j] = tmp_z_train.ravel()
        if RegType == "OLS": tmp_beta = OLS(tmp_z_train, tmp_X_train)
        if RegType == "RIDGE": tmp_beta = Ridge(tmp_z_train, tmp_X_train, lamb)
        z_pred[:,j] = (X_test @ tmp_beta).ravel()
        z_fit[:,j] = (tmp_X_train @ tmp_beta).ravel()

    return z_train_cp, z_test_cp, z_fit, z_pred

def kFold(x, y, z, d, k=5, shuffle = False, RegType="OLS", lamb=0):
    """
    Cross-Validation
    --------------------------------
    Input
        x,y,z:        Variables for data from Generate Data
        d:            Polynomial degree for the feature matrix
        k:            Number of folds
        shuffle:      Bool, shuffle the design matrix(default:False)
        RegType:      "OLS" for Ordinary least squares(default)
                      "RIDGE" for Ridge
        lamb:         the lambda value if RegType="RIDGE"
    --------------------------------
    Returns
        z_train_cp: (train_size, k)
        z_test_cp:  (test_size, k)
        z_fit:      (train_size, k)
        z_pred      (test_size, k)
    --------------------------------
    TODO:
    """
    X_ = PolyDesignMatrix(x, y, d)

    if shuffle == True:
        np.random.seed(42)
        np.random.shuffle(X_.T)

    z_pred, z_fit, z_test_cp, z_train_cp = [],[],[],[]
    for i in range(1, k+1):
        train_index, test_index = foldIndex(z, i, k)
        X_train = X_[train_index]
        X_test = X_[test_index]

        X_train, X_test = scale_X(X_train, X_test)

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

    # Make them into arrays and fix the shape
    z_test_k = np.zeros((len(z_test_cp[0]), k))
    z_train_k = np.zeros((len(z_train_cp[0]), k))
    z_pred_k = np.zeros((len(z_pred[0]), k))
    z_fit_k = np.zeros((len(z_fit[0]), k))

    for i in range(k):
        z_test_k[:,i] = z_test_cp[i].ravel()
        z_train_k[:,i] = z_train_cp[i].ravel()
        z_pred_k[:,i] = z_pred[i].ravel()
        z_fit_k[:,i] = z_fit[i].ravel()

    return z_train_k, z_test_k, z_fit_k, z_pred_k

def regression(z, X, rType="OLS", lamb=0):
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.33)
    X_train_scl, X_test_scl = scale_X(X_train, X_test)

    if rType =="OLS": beta = OLS(z_train, X_train_scl)
    if rType =="RIDGE": beta = Ridge(z_train, X_train_scl, lamb)

    z_fit = X_train_scl @ beta
    z_pred = X_test_scl @ beta

    return z_train, z_test, z_fit, z_pred

def get_beta(x, y, z, d, rType="OLS", lamb=0):

    X = PolyDesignMatrix(x, y, d)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.33)
    X_train_scl, X_test_scl = scale_X(X_train, X_test)

    if rType =="OLS": beta, var_beta = OLS(z_train, X_train_scl, var=True)
    if rType =="RIDGE": beta, var_beta = Ridge(z_train, X_train_scl, lamb, var=True)

    conf_beta = 1.96*np.sqrt(var_beta)  # 95% confidence

    z_fit = X_train_scl @ beta
    z_pred = X_test_scl @ beta

    return beta, conf_beta

def optimal_model_degree(x, y, z, metrics_test, metrics_train, rType = "OLS", lamb = 0, quiet = True, info=""):

    mse_min = metrics_test[1].min()     # The lowest MSE
    at_ = metrics_test[1].argmin()      # The index of mse_min
    best_degree = at_+1                 # The coresponding polynomial degree

    # Find the regression parameters for best_degree
    beta, conf_beta = get_beta(x, y, z, best_degree, rType, lamb)


    # The corresponding statistics:
    m_test_best = metrics_test[:,at_]
    m_train_best = metrics_train[:,at_]

    if quiet==False:
        print("Optimal model is")
        print ("    Deg  : {}".format(best_degree))
        print ("    RS2  : {:.3f} (train: {:.3f})".format(m_test_best[0], m_train_best[0]))
        print ("    MSE  : {:.3f} (train: {:.3f})".format(m_test_best[1], m_train_best[1]))
        print ("    Var  : {:.3f} (train: {:.3f})".format(m_test_best[2], m_train_best[2]))
        print ("    Bias : {:.3f} (train: {:.3f})".format(m_test_best[3], m_train_best[3]))
        print ("    Beta :", np.array_str(beta.ravel(), precision=2, suppress_small=True))
        print ("    Conf :", np.array_str(conf_beta.ravel(), precision=2, suppress_small=True))
        print("")
        plot.beta_conf(beta, conf_beta, best_degree, mse_min, m_test_best[0], rType, lamb, info)

    return beta, best_degree, m_test_best

def optimal_model_lamb(x, y, z, metrics_test, metrics_train, d, lambdas, rType = "RIDGE", quiet = True, info=""):

    mse_min = metrics_test[1].min()     # The lowest MSE
    at_ = metrics_test[1].argmin()      # The index of mse_min
    best_lamb = lambdas[at_]            # The corresponding lambda

    r2_best = metrics_test[0][at_]

    # Find the regression parameters for best_lamb
    beta, conf_beta = get_beta(x, y, z, d, rType, best_lamb)

    # The corresponding statistics:
    m_test_best = metrics_test[:,at_]
    m_train_best = metrics_train[:,at_]

    if quiet==False:
        print("Optimal model is")
        print ("    Deg  : {}".format(d))
        print ("    Lamb : {}".format(best_lamb))
        print ("    RS2  : {:.3f} (train: {:.3f})".format(m_test_best[0], m_train_best[0]))
        print ("    MSE  : {:.3f} (train: {:.3f})".format(m_test_best[1], m_train_best[1]))
        print ("    Var  : {:.3f} (train: {:.3f})".format(m_test_best[2], m_train_best[2]))
        print ("    Bias : {:.3f} (train: {:.3f})".format(m_test_best[3], m_train_best[3]))
        print ("    Beta :", np.array_str(beta.ravel(), precision=2, suppress_small=True))
        print ("    Conf :", np.array_str(conf_beta.ravel(), precision=2, suppress_small=True))
        print("")
        plot.beta_conf(beta, conf_beta, d, mse_min, m_test_best[0], rType="RIDGE", lamb = best_lamb, info = info)

    return beta, best_lamb, m_test_best

def map_to_data(mapdat):
    rows, columns = np.shape(mapdat)
    factor = max(rows, columns)
    xdim = np.linspace(0,columns,columns)/factor
    ydim = np.linspace(0,rows,rows)/factor
    x, y = np.meshgrid(xdim,ydim)
    x = np.array([x.ravel()]).T
    y = np.array([y.ravel()]).T
    z = np.array([mapdat.ravel()]).T
    return  [x,y], z, factor

if __name__ == '__main__':
    x, y, z = GenerateData(100, 0.01)
    degrees = np.arange(1, 10+1)

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
    X = PolyDesignMatrix(x, y, 2)

    # Test if the returned beta's are the same for small lambdas:
    beta_ridge, var_ridge = Ridge(z, X, 0.0000000001, True)
    beta_ols, var_ols = OLS(z,X, True)

    print("Ridge: ", np.array_str(beta_ridge.ravel(), precision=2, suppress_small=True))
    print("OLS:   ", np.array_str(beta_ols.ravel(), precision=2, suppress_small=True))

    print("Ridge: ", np.array_str(var_ridge.ravel(), precision=2, suppress_small=True))
    print("OLS:   ", np.array_str(var_ols.ravel(), precision=2, suppress_small=True))
    """
