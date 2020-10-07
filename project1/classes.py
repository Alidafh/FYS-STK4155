#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 13:39:08 2020

@author: gert
"""

import numpy as np
import functions as func
from sklearn.model_selection import train_test_split

class ML_Analysis:
    
    def __init__(self, x, z, dm_func = func.PolyDesignMatrix2, dm_params = 3):
        
        self.x = x                              # Raw data, independent variable, in the form [a, b, ...], where 
                                                # a,b,... are arrays of datapoints for the corresponding design parameters. 
                                        
        self.z = z                              # Raw data, response variable.
        
        self.dm_func = dm_func                  # Function that generates the design matrix
        
        self.dm_params = dm_params              # Arguments (other than x) that need to go into DM_func
        
        self.dm = self.generate_dm()             # The Design Matrix
        
        self.reg_func = func.OLS                # The function that defines the regression procedure
        
        self.reg_params                         # Extra arguments that are needed by reg_func 
        
        self.scale_func = func.scale_X          # The function that defines the scaling procedure
        
        self.scale_params = None                # Extra arguments that are needed by scale_func 
        
        self.split_func = train_test_split      # The function that defines the splitting procedure
        
        self.split_params = None                # Extra arguments that are needed by split_func 
        
        
        
    def generate_dm(self):
        
        return self.dm_func(self.x, self.dm_params)
    
    
    

class Bootstrap_Analysis(ML_Analysis):
    
    def __init__(self, x, z, dm_func = func.PolyDesignMatrix2, dm_params = 3,
                 n_iterations = 100):
        
        super().__init__(x, z, dm_func = dm_func, dm_params = dm_params)
        
        self.n_iterations = n_iterations
        
    
        
    def split(self)
        
        
    def iterate(self):
        
        for i in np.arange(n_iterations):
            
            tmp_X_train, tmp_z_train = resample(X_train, z_train)
            tmp_beta = func.OLS(tmp_z_train, tmp_X_train)
            z_pred[:,j] = X_test_scl @ tmp_beta.ravel()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        