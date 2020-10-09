import numpy as np

def SVDinv(A):
    """
    Credit: Morten Hjort-Jensen
    Takes as input a numpy matrix A and returns inv(A) based on singular value
    decomposition (SVD is numerically more stable than the inversion algorithms
    provided by numpy and scipy.linalg at the cost of being slower.
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
    Generates the indices of fold i of k
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
        X_test = X[test_index]
        X_train = X[train_index]
        print(X_test.shape, z_test.shape)
        print(X_train.shape, z_train.shape)
