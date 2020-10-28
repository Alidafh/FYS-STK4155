#!/usr/bin/python
import numpy as np
from sklearn.preprocessing import StandardScaler

def SVDinv(A):
    """
    Takes as input a numpy matrix A and returns inv(A) based on singular value
    decomposition (SVD is numerically more stable than the inversion algorithms
    provided by numpy and scipy.linalg at the cost of being slower.
    --------------------------------
    Input
        A: The matrix to invert
    --------------------------------
    Returns
        invA: The inverted matrix
    --------------------------------
    Credit: Morten Hjort-Jensen
    """
    U, s, VT = np.linalg.svd(A)
    D = np.zeros((len(U),len(VT)))
    for i in range(0,len(VT)):
        D[i,i]=s[i]
    UT = np.transpose(U)
    V = np.transpose(VT)
    invD = np.linalg.inv(D)
    invA = np.matmul(V,np.matmul(invD,UT))
    return invA

def foldIndex(dataset,i, k):
    """
    Generates the indices of fold i of k for the k-fold routine
    --------------------------------
    Input
        dataset: the dataset
        i: the number of the loop
        k: the
    --------------------------------
    Returns
        train_index: The index numbers for the training data
        test_index: The index numbers for the test data
    """
    n = len(dataset)
    indices = np.arange(n)
    a = n*(i-1)//k
    b = (n*(i)//k)
    test_index = indices[a:b]
    size_test = test_index.size
    size_train = int(np.abs(n - size_test))
    train_index = np.zeros(int(np.abs(n - size_test)), dtype=int)
    train_index[:a] = indices[:a]
    train_index[a:] = indices[b:]
    return train_index, test_index

def scale_X(train, test, scaler="manual"):
    """
    Scales the training and test data either by subtracting the mean and
    dividing by the std manually or using sklearn's StandardScaler.
    --------------------------------
    Input
        train: The training set
        test:  The test set
        scaler: Choose what scaler you want to use ("manual" or "sklearn")
    --------------------------------
    Returns
        train_scl: The scaled training set
        test_scl:  The scaled test set
    """

    if scaler == "sklearn":
        scaler = StandardScaler()
        scaler.fit(train[:,1:])
        train_scl =  np.ones(train.shape)
        test_scl = np.ones(test.shape)
        train_scl[:,1:] = scaler.transform(train[:,1:])
        test_scl[:,1:] = scaler.transform(test[:,1:])

    if scaler=="manual":
        train_scl =  np.ones(train.shape)
        test_scl = np.ones(test.shape)
        mean_train = np.mean(train[:,1:])
        std_train = np.std(train[:,1:])
        train_scl[:,1:] = (train[:,1:] - mean_train)/std_train
        test_scl[:,1:]= (test[:,1:] - mean_train)/std_train

    return train_scl, test_scl

def GenerateDataFranke(ndata, noise_str=0.1):
    """
    Generates data using the Franke Function with normally distributed noise
    of strength noise_str
    --------------------------------
    Input
        ndata: the number of datapoints you want
        noise_str: the strength of the noise, default is zero
    --------------------------------
    Returns
        input_features: the x1 and x2 values, shape (400, 2)
        y: Franke function values using the input features, shape (400,)
    """
    def FrankeFunction(x,y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        z = term1 + term2 + term3 + term4
        return z

    sqrtndata= int(round(np.sqrt(ndata)))
    np.random.seed(42)
    x1 = np.linspace(0,1, sqrtndata)
    x2 = np.linspace(0,1, sqrtndata)
    x1, x2 = np.meshgrid(x1, x2)
    y = FrankeFunction(x1, x2).ravel()
    noise = noise_str*np.random.normal(0, 1, y.shape)
    y = y + noise
    input_features = np.c_[x1.ravel(), x2.ravel()]
    return input_features, y

def GenerateDataLine(ndata=100):
    """
    Generates data for straight line regression
    --------------------------------
    Input
        ndata: the number of datapoints you want
    --------------------------------
    Returns
        X: The design matrix
        y: Datapoints
    """
    np.random.seed(42)
    x = 2*np.random.rand(ndata,1)
    y = 4+3*x+np.random.randn(ndata,1)
    X = np.c_[np.ones((ndata,1)), x]
    return X, y.ravel()

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
    """
    n = len(x)
    p = int(((d+2)*(d+1))/2)  # number of terms in beta
    X = np.ones((n, p))       # Matrix of size (n,p) where all entries are zero

    # fill colums of X with the polynomials [1  x  y  x**2  y**2  xy ...]
    for i in range(1, d+1):
        j = int(((i)*(i+1))/2)
        for k in range(i+1):
            X[:,j+k] = x**(i-k)*y**(k)
    return X

def learning_schedule(t, t0, t1):
    return t0/(t+t1)


if __name__ == '__main__':
    input, y = GenerateDataFranke(10, 0.1)
