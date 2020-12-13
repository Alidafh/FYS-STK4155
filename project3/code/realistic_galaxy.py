#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:31:48 2020

@author: gert
"""

from astropy.io import fits
import fitsio
import numpy as np
from matplotlib.colors import LogNorm
import tarfile
import os

section_width = 10
section_height = 10

# Extract the file if it hasn't already been extracted
if not os.path.isfile("fits/IEM_base_v2.fits"):
    print("Extracting")
    tar = tarfile.open("fits/IEM_base_v2.tar.xz", "r:xz")
    tar.extractall("fits/")
    tar.close()



data_dge = fits.open("fits/IEM_base_v2.fits")
data_j = fits.open("fits/j-factor-map-image.fits")
data_gc = fitsio.read("fits/map_model2_GC.fits")


map_dge = (data_dge[0].data[0,:,:]/np.max(data_dge[0].data[0,:,:])).clip(min=0)
map_j = (data_j[0].data/np.max(data_j[0].data)).clip(min=0)
map_gc = (data_gc/np.sum(data_gc)).clip(min=0)



dpp_dge = 0.125
dpp_j = 0.05725971370143149
dpp_gc = 0.4

center_dge_x = 1440.5
center_dge_y = 97
center_j = 734
center_gc = 50.5

np_dge_width = section_width/dpp_dge
np_dge_height = section_height/dpp_dge

class dge():
    def __init__(self, dim = (28,28,20), scale=1):
        self.dim = dim
        self.scale = scale
        self.dpp_x = section_width/self.dim[1]
        self.dpp_y = section_height/self.dim[0]
        self.np_width = section_width/self.dpp_x
        self.np_height = section_height/self.dpp_y
        self.margin_x = int(np_dge_width/self.np_width)
        self.margin_y = int(np_dge_height/self.np_height)

    def func(self,y):

        # print(E)
        row = np.zeros(self.dim[1])
        distance_y = y*self.dpp_y
        # distance_x = x*self.dpp_x
        mid_y = int(center_dge_y - distance_y/dpp_dge)
        
        
        for i, el in enumerate(row):

            distance_x = (i - (self.dim[1]+1)/2 ) * self.dpp_x
            mid_x = int(center_dge_x + distance_x/dpp_dge)
            row[i] = np.mean(map_dge[mid_y-self.margin_y:mid_y+self.margin_y+1, mid_x-self.margin_x:mid_x+self.margin_x+1])

        return row*self.scale



class dark_matter_profile():
    def __init__(self, dim = (28,28,20), scale=1):
        self.dim = dim
        self.scale = scale
        self.dpp_x = section_width/self.dim[1]
        self.dpp_y = section_height/self.dim[0]
        self.middle_row = int((self.dim[0]+1)/2)
        self.middle_col = int((self.dim[1]+1)/2)
    def func(self, i,j):
        distance_y = (self.middle_row-i)*self.dpp_y
        distance_x = (self.middle_col-j)*self.dpp_x
        j = map_j[int(center_j - distance_y/dpp_j), int(center_j+distance_x/dpp_j)]

        return j*self.scale


class galactic_center_spectrum():
    def __init__(self, scale=1, k0 = 254.999992475857e-20, E0=1e+2, Ecut=10.6999998092651e+2, gamma=-2.14000010490417):
        self.k0 = k0
        self.E0=E0
        self.Ecut=Ecut
        self.gamma=gamma
        self.scale = scale
    def func(self,E):
        E = E+4
        return self.scale * self.k0 * ( E/self.E0 )**self.gamma * np.exp( -E/self.Ecut )


class galactic_center_profile():
    def __init__(self, dim = (28,28,20), scale=1):
        self.dim = dim
        self.scale = scale
        self.dpp_x = section_width/self.dim[1]
        self.dpp_y = section_height/self.dim[0]
        self.middle_row = int((self.dim[0]+1)/2)
        self.middle_col = int((self.dim[1]+1)/2)
    def func(self, i,j):
        distance_y = (self.middle_row-i)*self.dpp_y
        distance_x = (self.middle_col-j)*self.dpp_x
        flux = map_gc[int(center_gc - distance_y/dpp_gc), int(center_gc+distance_x/dpp_gc)]
        return flux*self.scale
    

