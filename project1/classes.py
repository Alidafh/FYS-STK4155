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
        
        
    def generate_dm(self):
        
        return self.dm_func(self.x, self.dm_params)
    
    
    def split(self):
        
        self.dm_train, self.dm_test, self.z_train, self.z_test = self.split_func(self.dm, self.z, test_size = self.test_size)
        return self.dm_train, self.dm_test, self.z_train, self.z_test
    
    
    def scale(self):
        
        self.dm_train, self.dm_test = self.scale_func(self.dm_train, self.dm_test)
        return self.dm_train, self.dm_test
    
    
    
    
    
    

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
        
        
        
        
        