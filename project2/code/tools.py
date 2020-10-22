#!/usr/bin/python
import numpy as np

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


if __name__ == '__main__':
    import functions as func
    x, y, z = func.GenerateData(100, 0.1)
    X = func.PolyDesignMatrix(x, y, 2)
    k = 6

    for i in range(1, k+1):
        print("------------")
        print("i=", i)
        print("------------")
        train_index, test_index = foldIndex(z, i, k)
        #print(train_index, test_index)
        z_test = z[test_index]
        z_train = z[train_index]
