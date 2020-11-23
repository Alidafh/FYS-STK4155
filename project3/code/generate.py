#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def func(x):
    return 2*x

def DM_profile(r):
    # return 1/(r+0.1)
    return max(  10 - r/2.5  , 0 )


def DM_spectrum(E):

    sigma = 1
    mean = 5
    norm = 2

    return norm*np.exp(- ( (E - mean)**2 )/(2*sigma**2) )


def galaxy_spectrum(E):

    norm = 1
    grad = 0.1

    return norm - grad*E

#def func(x):
#    return np.exp(0.1*x)

class SkyMap:
    def __init__(self, dim):
        self.dim = dim
        #print(dim)
        self.matrix = np.zeros(self.dim)




    def generate_galaxy(self, noise):

        galaxy = np.zeros(self.dim)
        middle_row = int(self.dim[0]/2)

        for i in range(self.dim[0]):    # loop over rows

            for E in range(self.dim[2]):

                if i == middle_row:
                    galaxy[middle_row,:,E] = func(middle_row)*np.ones(self.dim[1])*galaxy_spectrum(E)
                if i < middle_row:
                    galaxy[i,:,E] = func(i)*np.ones(self.dim[1])*galaxy_spectrum(E)
                if i > middle_row:
                    x = np.abs(i-2*middle_row)
                    galaxy[i,:,E] = func(x)*np.ones(self.dim[1])*galaxy_spectrum(E)

        noise_ = noise*np.ones(self.dim)
        galaxy = galaxy+noise_

        self.matrix += galaxy

        return galaxy


    def generate_DM(self, noise):
        dark_matter = np.zeros(self.dim)
        middle_row = int(self.dim[0]/2)
        middle_col = int(self.dim[1]/2)

        for i in range(self.dim[0]):
            for j in range(self.dim[1]):

                r = np.sqrt(   np.abs(middle_row-i)**2 + np.abs(middle_col-j)**2)

                for E in range(self.dim[2]):

                    dark_matter[i,j,E] = DM_profile(r)*DM_spectrum(E)

        self.matrix += dark_matter
        return dark_matter


    def generate_galaxy_noise(self, noise):

        noise = noise*np.random.randn(self.dim[0],self.dim[1], self.dim[2])

        self.matrix += noise

        return noise


    def generate_DM_noise(self, noise):

        noise = noise*np.random.randn(self.dim[0],self.dim[1], self.dim[2])

        self.matrix += noise
        return noise


        return noise



    def write_to_file(self):
        return 0



    def read_from_file(self):
        return 0


    def ravel_map(self, matrix):
        b = matrix.ravel()
        self.dim_ravel= b.shape
        return b



    def unravel_map(self, data):
        if len(data.shape)==1:
            d = data.reshape(self.dim)
        else:
            d = data
        return d


    def combine_slices(self, data):
        data_ = self.unravel_map(data)
        data_combined = np.sum(data_, axis=2)
        return data_combined



    def display(self, data, slice=None, save_as=None):
        """
        Display the map. If no energy slice option is chosen, the energy levels
        are added together. Save figure
        """
        if slice:
            data_ = self.unravel_map(data)
            data_ = data_[:,:,slice]
        else:
            data_= self.combine_slices(data)

        y_axis = data_.shape[0]/2
        x_axis = data_.shape[1]/2
        axis_range = [-x_axis,x_axis,-y_axis, y_axis]

        fig = plt.figure()
        plt.xlabel('Galactic Longitude')
        plt.ylabel('Galactic Latitude')

        ax = plt.gca()
        #https://imagine.gsfc.nasa.gov/science/toolbox/gamma_generation.html
        im = ax.imshow(data_, cmap="inferno", extent = axis_range)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        if save_as: fig.savefig(save_as)


    def display_spectrum(self):

        plt.figure(50)

        mat = self.matrix

        spectrum = np.sum(mat, axis = (0,1))

        plt.plot(spectrum)

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

def main_gert():

    E = 4

    map = SkyMap(dim=(50,100,10))
    gal = map.generate_galaxy(noise=0.1)[:,:,E]
    galn = gal + map.generate_galaxy_noise(noise=2)[:,:,E]
    dm = map.generate_DM(noise=0.1)[:,:,E]
    dmn = dm + map.generate_DM_noise(noise=1)[:,:,E]
    fig, ax = plt.subplots(nrows=1, ncols=3,  figsize=(10, 3))
    ax[0].imshow(gal)
    ax[1].imshow(dm)
    ax[2].imshow(dmn+galn)
    map.display_spectrum()
    plt.show()

def main0():
    PATH="../data/"
    generate_data(nMaps=5, dim=(50,100,10), PATH=PATH)

def main1():
    PATH="../data/"
    galaxy, dim = read_data(PATH, filename="galaxy_(50, 100, 10)_.csv")
    dm, dim = read_data(PATH, filename="DM_(50, 100, 10)_.csv")

    map1 = SkyMap(dim)
    map1.display(dm[0], slice=0, save_as="../figures/test.png")
    plt.show()


if __name__ == '__main__':
    #main_gert()
    main0()
    main1()

"""
    fig, ax = plt.subplots(nrows=1, ncols=3,  figsize=(10, 3))
    im0 = ax[0].imshow(data_[:,:,0], vmin=0, vmax=250)
    fig.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(data_[:,:,-1], vmin=0, vmax=250)
    fig.colorbar(im1, ax=ax[1])
    im2 = ax[2].imshow(data_combined,vmin=0, vmax=250)
    fig.colorbar(im2, ax=ax[2])
    plt.show()



    fig = plt.figure(figsize=(10,3))
    fig.add_subplot(121)
    map1.display(galaxy_to_display, slice=0)
    fig.add_subplot(122)
    map1.display(galaxy_to_display)
    plt.show()
"""
