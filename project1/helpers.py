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
