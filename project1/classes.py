#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 13:39:08 2020

@author: gert
"""

import numpy as np
import functions as func
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import tools
import os, errno
import matplotlib.pyplot as plt

class ML_Analysis:
    
    def __init__(self, x, z, dm_func = func.PolyDesignMatrix2, dm_params = 3):
        
        self.x = x                              # Raw data, independent variable, in the form [a, b, ...], where 
                                                # a,b,... are arrays of datapoints for the corresponding design parameters. 
                                        
        self.z = z                              # Raw data, response variable.
        
        self.dm_func = dm_func                  # Function that generates the design matrix. Form has to be dm_func([x,y], params)
        
        self.dm_params = dm_params              # Arguments (other than x) that need to go into DM_func
        
        self.dm = self.generate_dm()            # The Design Matrix
        
        self.reg_func = func.OLS2               # The function that defines the regression procedure. Has to be on the form
                                                # reg_func(z, dm, reg_params).
        
        self.reg_params = None                  # Extra arguments that are needed by reg_func 
        
        self.scale_func = func.scale_X          # The function that defines the scaling procedure
        
        self.scale_params = None                # Extra arguments that are needed by scale_func 
        
        self.split_func = train_test_split      # The function that defines the splitting procedureHas to be on the form
                                                # split_func(dm, z, test_size = test_size)
                                                
        self.split_params = None                # Extra arguments that are needed by split_func 
        
        self.test_size = 0.3                    # Proportion of data that should be testing data
        
        self.dm_train = None                    # Set of training data; design matrix 
        
        self.dm_test = None                     # Set of testing data; design matrix 
        
        self.z_train = None                     # Set of training data; response variable
        
        self.z_test = None                      # Set of testing data; response variable
        
        self.r2_score = None                    # R2 score of the resulting fit. 
        
        self.mse = None                         # mse score of the resulting fit.
        
        self.var = None                         # var score of the resulting fit.
        
        self.bias = None                        # bias score of the resulting fit.
        
        self.beta = None                        # Vector of optimal coefficients
        
        #self.original_image = None              # Matrix representing the original image from which the data is generated.
        
        #self.image_dims = None                  # Dimensions of the original image. 
        
        self.z_pred = None                      # Data predicted based on fitted parameters
        
        #self.image_pred = None                  # Matrix form of predicted data. 
        
        #self.original_image_test = None         #The original image, including only the pixels corresponding to test data. 
        
        
        
    def generate_dm(self):
        
        return self.dm_func(self.x, self.dm_params)
    
    
    def split(self):
        
        self.dm_train, self.dm_test, self.z_train, self.z_test = self.split_func(self.dm, self.z, test_size = self.test_size)
        return self.dm_train, self.dm_test, self.z_train, self.z_test
    
    
    def scale(self):
        
        self.dm_train, self.dm_test = self.scale_func(self.dm_train, self.dm_test)
        return self.dm_train, self.dm_test


    def check_original_image(self):
        
        if self.original_image != None:
            rows = np.shape(self.original_image)[0]
            columns = np.shape(self.original_image)[1]
            self.image_dims =  [rows, columns]
            
            
    def generate_prediction(self):
        
        self.z_pred = self.dm_test @ self.beta
        #self.check_original_image()
        #self.image_pred = self.z.reshape(self.image_dims[0], self.image_dims[1])
        
        return self.z_pred  #, self.image_pred
    
    
    def fit(self):
        
        self.generate_dm()
        self.split()
        self.scale()
        self.beta = self.reg_func(self.dm_train, self.z_train, self.reg_params)
        self.generate_prediction()
        
        return self.beta
    
    

class Bootstrap_Analysis(ML_Analysis):
    
    
    
    def __init__(self, x, z, dm_func = func.PolyDesignMatrix2, dm_params = 3,
                 n_iterations = 100):
        
        super().__init__(x, z, dm_func = dm_func, dm_params = dm_params)
        
        self.n_iterations = n_iterations
        
        
        
    def iterate(self):
        
        z_pred = np.empty((self.z_test.shape[0], self.n_iterations))
        for j in np.arange(self.n_iterations):
            
            tmp_X_train, tmp_z_train = resample(self.dm_train, self.z_train)
            tmp_beta = self.reg_func(tmp_X_train, tmp_z_train, self.reg_params)
            z_pred[:,j] = self.dm_test @ tmp_beta.ravel()
        
        self.r2_score, self.mse, self.var, self.bias = func.metrics(self.z_test, z_pred, test=True)
        
        return self.r2_score, self.mse, self.var, self.bias 
    
    
    
    def analysis(self):
        
        self.generate_dm()
        self.split()
        self.scale()
        self.iterate()
        
        
        
        
        
        
        

class CV_Analysis(ML_Analysis):
    
    
    
    def __init__(self, x, z, dm_func = func.PolyDesignMatrix2, dm_params = 3,
                 k = 5):
        
        super().__init__(x, z, dm_func = dm_func, dm_params = dm_params)
        
        self.k = k
            
        self.mse_k = np.zeros((self.k))       
        
        self.bias_k = np.zeros((self.k))       
        
        self.r2_score_k = np.zeros((self.k))        
       
        self.var_k = np.zeros((self.k))  




    def iterate(self):
            
    
        np.random.seed(42)
        fold_i = 0
        for i in range(1, self.k+1):
            """loop over folds and calculate the fitted and predicted z values"""
            train_index, test_index = tools.foldIndex(self.x[0], i, self.k)
            self.dm_train = self.dm[train_index]
            self.z_train = self.z[train_index]

            self.dm_test = self.dm[test_index]
            self.z_test = self.z[test_index]

            self.dm_train, self.dm_test = self.scale()

            self.beta = self.reg_func(self.dm_train, self.z_train, self.reg_params)
            
            # z_fit = X_train_scl @ beta
            z_pred = self.dm_test @ self.beta
            
            self.r2_score_k[fold_i], self.mse_k[fold_i], self.var_k[fold_i], self.bias_k[fold_i]= func.metrics(self.z_test, z_pred, test=True)
            
           
            fold_i +=1
        
        self.r2_score = np.mean(self.r2_score_k)
        self.mse = np.mean(self.mse_k)
        self.bias = np.mean(self.bias_k)
        self.var = np.mean(self.var_k)
    
    
        
    
    def analysis(self):
        
        self.generate_dm()
        self.iterate()
        
        
        

class Plot_Project1:
    
    
    def __init__(self, direc = "output/figures_gert"):
        
        
        self.direc = direc
        
        self.title = None

       

        if not os.path.exists(os.path.dirname(self.direc)):
            try:
                os.makedirs(os.path.dirname(self.direc))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise






class Iterate_Lambda(Plot_Project1):
    
    
    def __init__(self, lambda_range = [-3, 1, 0.5], range_type = "log", 
                 reg_func = func.OLS2):
        
        super().__init__()
        
        
        self.subdirec = "lambda"
    
        self.xlabel = "$$\lambda$$"
        
        self.ylabel = "MSE"
        
        self.lambda_range = lambda_range
        
        self.range_type = range_type
        
        self.ns = [100]
        
        self.ds = [3,5,7]
        
        self.tps = [0.3] 
    
        self.reg_func = reg_func
        
        self.reg_name = "OLS"
    
        self.mse = None
        
        self.bias = None
        
        self.var = None
        
        self.r2_score = None
        
        self.i_min = -1
                            
        self.c_min = -1
                            
        self.m_min = -1
                            
        self.s_min = -1
        
        self.plot_all_n = 1
        
        self.plot_all_d = 1
        
        self.plot_all_tp = 1
        
        self.plot_log = 1
    
        if not os.path.exists(os.path.dirname(self.direc + "/" + self.subdirec + "/")):
            try:
                os.makedirs(os.path.dirname(self.direc + "/" + self.subdirec +"/"))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                
    
    
    def iterate(self):
        
        lam_vec = np.arange(self.lambda_range[0], self.lambda_range[1], self.lambda_range[2])
        
        
        if self.range_type == "log":
            lam_vec = 10**lam_vec
        elif self.range_type == "lin":
            lam_vec = lam_vec
        else:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!        ERROR: lambda_range is invalid!         !!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        self.lam_vec = lam_vec
        
        self.mse = np.zeros((len(lam_vec),len(self.ns), len(self.ds), len(self.tps)))
        self.bias = np.zeros((len(lam_vec),len(self.ns), len(self.ds), len(self.tps)))
        self.var = np.zeros((len(lam_vec),len(self.ns), len(self.ds), len(self.tps)))
        self.r2_score = np.zeros((len(lam_vec),len(self.ns), len(self.ds), len(self.tps)))
        
        mse_min = float('inf')
        
        
        for m, n in enumerate(self.ns):
            
            x, y, z = func.GenerateData(n, 0.01, "debug")
        
            bs = Bootstrap_Analysis([x,y], z)
            bs.reg_func = self.reg_func
                     
            
            for c, d in enumerate(self.ds):
                
                bs.dm_params = d
                
                
                for s, tp in enumerate(self.tps):
                    
                    bs.test_size = tp
        
        
                    for i, lam in enumerate(lam_vec):
                        
                        
                        bs.reg_params = lam
                        bs.analysis()
                        
                        self.mse[i,m,c,s] = bs.mse
                        self.bias[i,m,c,s]  = bs.bias
                        self.var[i,m,c,s]  = bs.var
                        self.r2_score[i,m,c,s]  = bs.r2_score
                        
                        if self.mse[i,m,c,s] < mse_min: 
                            mse_min = self.mse[i,m,c,s]
                            self.i_min = i
                            self.c_min = c
                            self.m_min = m
                            self.s_min = s
                
    

    
    def print_results(self):
        
        print(" \"Optimal\" MSE: \t \t {:}".format(self.mse[self.i_min,self.m_min,self.c_min,self.s_min]))
        print(" \"Optimal\" Bias: \t \t {:}".format(self.bias[self.i_min,self.m_min,self.c_min,self.s_min]))
        print(" \"Optimal\" Variance: \t {:}".format(self.var[self.i_min,self.m_min,self.c_min,self.s_min]))
        print(" \"Optimal\" R2-score: \t {:}".format(self.r2_score[self.i_min,self.m_min,self.c_min,self.s_min]))
        print("")
        if self.range_type == "log":
            print(" \"Optimal\" Lambda: \t\t\t\t\t {:}".format(10**self.lam_vec[self.i_min]))
        elif self.range_type == "lin":
            print(" \"Optimal\" Lambda: \t\t\t\t\t {:}".format(self.lam_vec[self.i_min]))
        print(" \"Optimal\" Number of data points: \t\t {:}".format(self.ns[self.m_min]))
        print(" \"Optimal\" Number of degrees: \t\t {:}".format(self.ds[self.c_min]))
        print(" \"Optimal\" Proportion of test data: \t {:}".format(self.tps[self.s_min]))
        
    
    def plot_results(self):
        
        
        for m, n in enumerate(self.ns):
    
            
            fig = plt.figure("n{:}".format(n))
            plt.grid()
            plt.title("{:} with {:} data points".format(self.reg_name, n), fontsize = 12, fontname = "serif")
            plt.xlabel("lambda", fontsize = 12, fontname = "serif")
        
            plt.plot(self.lam_vec, self.mse[:,m,self.c_min,self.s_min], "tab:red", label="MSE")
            plt.plot(self.lam_vec, self.var[:,m,self.c_min,self.s_min], "tab:blue", label="Variance")
            plt.plot(self.lam_vec, self.bias[:,m,self.c_min,self.s_min], "tab:green", label="Bias")
            plt.legend()
            if self.plot_log==1: plt.semilogy()
        
            
            if not os.path.exists(os.path.dirname(self.direc + "/" + self.subdirec + "/" + "n/")):
                try:
                    os.makedirs(os.path.dirname(self.direc + "/" + self.subdirec + "/" + "n/"))
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        raise
            
            
            fig.savefig("{}/{}/n/n{}.png".format(self.direc,self.subdirec,n))
            
            
            file = open("{}/{}/n/n{}.txt".format(self.direc,self.subdirec,n), "w+")
            file.write(" \"Optimal\" Number of data points: \t\t {:}".format(self.ns[self.m_min]))
            file.close
        
        
        
    def analysis(self):
        
        self.iterate()
        self.print_results()
        self.plot_results()
        
    
    
    
test = Iterate_Lambda()
test.ns =[100, 1000, 10000]   
test.analysis()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        