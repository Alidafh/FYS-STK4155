#!/usr/bin/python
import numpy as np
import pandas as pd
from imageio import imread
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import functions as func
import plotting as plot
import tools as tools
import sys
#!/usr/bin/python
import numpy as np
import pandas as pd
from imageio import imread
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import functions as func
import plotting as plot
import tools as tools
import sys

x, y, z = func.GenerateData(100, 0.01, "debug")

###############################################################################
def part_a(x, y, z, degree=5):
    print ("------------------------------------------------------")
    print ("                      PART A                          ")
    print ("------------------------------------------------------")

    plot.plot_franke("Illustration of the Franke Function", "franke_illustration", 0.1)

    X = func.PolyDesignMatrix(x, y, degree)

    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.33)

    X_train_scl, X_test_scl = func.scale_X(X_train, X_test)

    print("Fitting with OLS:")
    beta, var_beta = func.OLS_SVD(z_train, X_train_scl, var=True)
    conf_beta = 1.96*np.sqrt(var_beta)  # 95% confidence

    z_train_fit = X_train_scl @ beta
    z_test_pred = X_test_scl @ beta

    R2_train, MSE_train, var_train, bias_train = func.metrics(z_train, z_train_fit)
    R2_test, MSE_test, var_test, bias_test = func.metrics(z_test, z_test_pred)
    #print ("----------------------")
    print ("    Deg : {}".format(degree))
    print ("    RS2 : {:.3f}".format(R2_train))
    print ("    MSE : {:.3f}".format(MSE_train))
    print ("    Beta:", np.array_str(beta.ravel(), precision=2, suppress_small=True))
    print ("    Conf:", np.array_str(conf_beta, precision=2, suppress_small=True))
    print ("")
    #print ("----------------------")
    plot.plot_beta(beta.ravel(), conf_beta, degree)

part_a(x,y,z,3)

###############################################################################

def part_b_noresample(x, y, z, d=5):
    print ("------------------------------------------------------")
    print ("                      PART B                          ")
    print ("                   no resampling                      ")
    print ("------------------------------------------------------")

    print("Preforming OLS-regression using polynomials up to {} degrees\n".format(d))
    mse_test = np.zeros(d)
    mse_train = np.zeros(d)
    degrees = np.arange(1, d+1)

    for i in range(d):
        """ Loop over degrees"""
        X = func.PolyDesignMatrix(x, y, degrees[i])

        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.33)
        X_train_scl, X_test_scl = func.scale_X(X_train, X_test) # Scale data

        beta = func.OLS_SVD(z_train, X_train_scl)

        z_train_fit = X_train_scl @ beta
        z_test_pred = X_test_scl @ beta

        R2_train, MSE_train, var_train, bias_train = func.metrics(z_train, z_train_fit)
        R2_test, MSE_test, var_test, bias_test = func.metrics(z_test, z_test_pred)
        mse_train[i] = MSE_train
        mse_test[i] = MSE_test

    plot.plot_MSE(degrees, mse_test, mse_train, "OLS", "degrees_{}".format(d))

part_b_noresample(x,y,z,10)
###############################################################################

def part_b_bootstrap(x, y, z, d=5, n_bootstraps=100, write=False):
    print ("------------------------------------------------------")
    print ("                      PART B                          ")
    print ("                    resampling                        ")
    print ("------------------------------------------------------")

    print("Preforming OLS-regression using polynomials up to {:.0f} degrees with n_bootstrap={:.0f}\n".format(d, n_bootstraps))

    # Initialize arrays of shape (degrees, )
    bias = np.zeros(d)
    variance = np.zeros(d)
    mse = np.zeros(d)
    r2_score = np.zeros(d)

    degrees = np.arange(1, d+1) # array of degrees

    if write==True:
        f = open("output/outfiles/bootstrap_metrics_{:.0f}_{:.0f}.txt".format(len(z), n_bootstraps),"w+")
        print("Writing metrics to file:")
        print("   output/outfiles/bootstrap_metrics_{:.0f}_{:.0f}.txt\n".format(len(z), n_bootstraps))
        f.write("Polydegree   R2   MSE   Variance   Bias\n")

    for i in range(d):
        """ Loop over degrees"""
        X = func.PolyDesignMatrix(x, y, degrees[i])

        # Split and scale data
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.33)
        X_train_scl, X_test_scl = func.scale_X(X_train, X_test)

        z_test_pred = np.empty((z_test.shape[0], n_bootstraps))  # matrix shape (z_test, boostraps)
        for j in range(n_bootstraps):
            """ Loop over bootstraps"""
            tmp_X_train, tmp_z_train = resample(X_train, z_train)
            tmp_beta = func.OLS_SVD(tmp_z_train, tmp_X_train)
            z_test_pred[:,j] = X_test_scl @ tmp_beta.ravel()

        # Calculate the stuff
        r2_score[i], mse[i], variance[i], bias[i] = func.metrics(z_test, z_test_pred)
        if write==True:
            f.write("{:.0f}   {:.5e}  {:.5e}  {:.5e}  {:.5e}\n".format(degrees[i], r2_score[i], mse[i], variance[i], bias[i]))

    plot.bias_variance(degrees, mse, variance, bias, rType = "OLS", c = "degrees_{:.0f}_ndata_{:.0f}_nboot_{:.0f}".format(d, len(z), n_bootstraps))


part_b_bootstrap(x, y, z, d=10, n_bootstraps=100, write=True)

###############################################################################

def kFold(x, y, z, d=5, k=5, shuffle = False):
    """
    --------------------------------
    Input
    --------------------------------
    """
    print ("------------------------------------------------------")
    print ("                      PART C                          ")
    print ("                      k-fold                          ")
    print ("------------------------------------------------------")

    degrees = np.arange(1, d+1)

    mse_kFold = np.zeros((d,k))        # arrays of statistics  where each row
    bias_kFold = np.zeros((d,k))       # corresponds to a degree and each colunm
    rs2_kFold = np.zeros((d,k))        # is corresponds to the fold number
    var_kFold = np.zeros((d,k))

    a = 0
    for j in range(d):
        """loop over degrees"""
        degree = degrees[j]
        X = func.PolyDesignMatrix(x, y, degree)
        if shuffle == True: np.random.shuffle(X) # Shuffle the rows
        b = 0
        for i in range(1, k+1):
            """loop over folds"""
            train_index, test_index = tools.foldIndex(x, i, k)

            X_train = X[train_index]
            z_train = z[train_index]

            X_test = X[test_index]
            z_test = z[test_index]

            X_train_scl, X_test_scl = func.scale_X(X_train, X_test)

            beta = func.OLS_SVD(z_train, X_train_scl)

            z_fit = X_train_scl @ beta
            z_pred = X_test_scl @ beta

            rs2_kFold[a,b], mse_kFold[a,b], var_kFold[a,b], bias_kFold[a,b]= func.metrics(z_test, z_pred)

            b +=1
        a +=1

    plot.plot_kFold_var(degrees, mse_kFold, k, rType="OLS", varN="MSE")
    
    estimated_rs2_KFold = np.mean(rs2_kFold, axis = 1)
    estimated_mse_KFold = np.mean(mse_kFold, axis = 1)
    estimated_var_KFold = np.mean(var_kFold, axis = 1)
    estimated_bias_KFold = np.mean(bias_kFold, axis = 1)
    return estimated_rs2_KFold, estimated_mse_KFold, estimated_var_KFold, estimated_bias_KFold

kFold(x,y,z, d=5, k=5, shuffle=True)




############################# DO NOT ERASE ####################################
########################### (Without asking) ####################################

"""
scaler = StandardScaler()
scaler.fit(X_train[:,1:])
X_train_scaled = scaler.transform(X_train[:,1:])
#X_test_scaled = scaler.transform(X_test)

n = len(X_train_scaled[:,1])
ones = np.ones((n,1))

X_train_new = np.hstack((ones, X_train_scaled))

z_train_predict, beta, conf_inter_beta = func.OLS(z_train, X_train_new)
R2, MSE, var = func.metrics(z_train, z_train_predict)
print("---sklear--------")
print("R2: ", R2)
print("MSE: ", MSE)

"""

"""
terrain1 = imread("datafiles/SRTM_data_Norway_1.tif")

plt.figure()
plt.title("Terrain over Norway 1")
plt.imshow(terrain1, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
"""
