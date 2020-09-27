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

def GenerateDataOld(nData, noise_str=0, seed=""):
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

    x = np.arange(0, 1, 1./nData)
    y = np.arange(0, 1, 1./nData)
    x,y = np.meshgrid(x,y)

    z = FrankeFunction(x, y)
    if noise_str != 0:
        noise = noise_str * np.random.randn(nData, 1)
        z += noise

    return x, y, z

def foldIndex(dataset,i, k):
    """
    Generates the indices of fold i of k
    """
    n = len(dataset)
    indices = np.arange(n)
    train_s = n*(k-1)//k
    a = n*(i-1)//k
    b = (n*(i)//k)
    test_index = indices[a:b]
    train_index = np.zeros(int(train_s), dtype=int)
    train_index[:a] = indices[:a]
    train_index[a:] = indices[b:]
    return train_index, test_index
