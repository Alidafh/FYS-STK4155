import numpy as np
from sklearn.model_selection import train_test_split
from regression import Regression
from sklearn.utils import resample

class Resample(Regression):
    def Bootstrap(self, X, y, nbs, method):
        y_copy = np.zeros((y.shape[0], nbs))
        for i in range(nbs):
            tmp_X, tmp_y = resample(X, y)
            y_copy[:,i] = tmp_y.ravel()
        print(y_copy.shape)
        return 0




"""
def Bootstrap(x, y, z, d, n_bootstraps, RegType, lamb=0):

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
        if RegType == "LASSO": tmp_beta = lasso(tmp_z_train, tmp_X_train, lamb)
        z_pred[:,j] = (X_test @ tmp_beta).ravel()
        z_fit[:,j] = (tmp_X_train @ tmp_beta).ravel()

    return z_train_cp, z_test_cp, z_fit, z_pred

def kFold(x, y, z, d, k=5, shuffle = False, RegType="OLS", lamb=0):

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
        if RegType == "LASSO": beta = lasso(z_train, X_train, lamb)

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
"""
