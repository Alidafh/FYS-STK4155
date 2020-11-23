#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return 2*x

#def func(x):
#    return np.exp(0.1*x)

class SkyMap:
    def __init__(self, dim):
        self.dim = dim
        #print(dim)

    def generate_galaxy(self, noise):
        galaxy = np.zeros(self.dim)
        middle_row = int(self.dim[0]/2)

        for i in range(self.dim[0]):    # loop over rows
            if i == middle_row:
                galaxy[middle_row,:,0] = func(middle_row)*np.ones(self.dim[1])
            if i < middle_row:
                galaxy[i,:,0] = func(i)*np.ones(self.dim[1])
            if i > middle_row:
                x = np.abs(i-2*middle_row)
                galaxy[i,:,0] = func(x)*np.ones(self.dim[1])

        noise_ = noise*np.ones(self.dim)
        galaxy = galaxy+noise_

        return galaxy


    def generate_DM(self, noise):
        dark_matter = np.zeros(self.dim)
        middle_row = int(self.dim[0]/2)
        middle_col = int(self.dim[1]/2)

        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                if i == middle_row and j==middle_col:
                    dark_matter[i,j] = i*j
                    dark_matter[i-1, j-1] = i*j
                    dark_matter[i+1, j+1] = i*j
                    dark_matter[i-1, j] = i*j
                    dark_matter[i+1, j] = i*j
                    dark_matter[i, j-1] = i*j
                    dark_matter[i, j+1] = i*j


        return dark_matter

    def write_to_file(self):
        return 0

    def read_from_file(self):
        return 0

    def ravel_map(self, matrix):
        b = matrix.ravel()
        self.dim_ravel= b.shape
        return b

    def unravel_map(self, matrix_raveled):
        d = matrix_raveled.reshape(self.dim)
        return d

    def Display(self, data):
        if len(data.shape)==1:
            data_ = self.unravel_map(data)

            if len(data_.shape)>2:
                data_= data_[:,:,0]

        plt.imshow(data_)
        plt.show()


#=============================================================================

def generate_data(nMaps, dim, PATH):
    """
    Generates galaxies and saves them to a datafile. The galaxies are raveled
    and stored as a row in a numpy array of dimentions(nMaps, dim_ravel)
    ---------------
    Input:
        nMaps: The number of maps needed
        PATH: The path to where the data should be stored
    ---------------
    Returns:
        data: The data, shape(nMaps, N)
    """
    dim_ravel = np.prod(dim)    # dimentions of the raveled matrices
    galaxies = np.zeros((nMaps, dim_ravel))
    dark_matter = np.zeros((nMaps, dim_ravel))

    for i in range(nMaps):
        map = SkyMap(dim)

        galaxy = map.generate_galaxy(noise=0.1)
        dm = map.generate_DM(noise=0.1)

        galaxies[i,:] = map.ravel_map(galaxy)
        dark_matter[i,:] = map.ravel_map(dm)

    filename1 = "galaxy_{:}_".format(dim)
    filename2 = "DM_{:}_".format(dim)

    np.savetxt(PATH+filename1+".csv", galaxies, fmt="%.16f")
    np.savetxt(PATH+filename2+".csv", dark_matter, fmt="%.16f")
    return galaxies, dark_matter


def read_data(PATH, filename):
    """
    Reads the datafile created by the funtion generate_data
    ---------------
    Input:
        PATH: path to where the data is stored
        filename: the filename of the data
    ---------------
    returns:
        data: ndarray, shape (nMaps, N)
    """
    dim = filename.split("_")[1].strip("()").split(",")
    dim = [int( dim[i]) for i in range(len(dim))]

    data = np.loadtxt(PATH+filename)

    return data, dim

def main():
    map = SkyMap(dim=(50,100, 10))
    gal = map.generate_galaxy(noise=0.1)
    #dm = map.generate_DM(noise=0.1)
    fig, ax = plt.subplots(nrows=1, ncols=3,  figsize=(10, 3))
    ax[0].imshow(gal[:,:,0])
    #ax[1].imshow(dm)
    #ax[2].imshow(gal+dm)
    plt.show()


def main0():
    PATH="../data/"
    generate_data(nMaps=5, dim=(50,100,10), PATH=PATH)

def main1():
    PATH="../data/"
    data_galaxy, dim = read_data(PATH, filename="galaxy_(50, 100, 10)_.csv")
    data_dm, dim = read_data(PATH, filename="DM_(50, 100, 10)_.csv")

    map1 = SkyMap(dim)
    galaxy_to_display = data_galaxy[0]
    dm_to_display = data_dm[0]
    map1.Display(galaxy_to_display)
    map1.Display(dm_to_display)

if __name__ == '__main__':
    #main0()
    main1()
