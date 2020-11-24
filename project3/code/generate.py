#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.utils import shuffle
import sys



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
        self.matrix = np.zeros(self.dim)


    def set_matrix(self, matrix):
        """ set the self.matrix attribute """
        matrix = self.unravel_map(matrix)
        self.matrix = matrix


    def check_matrix(self):
        """ check if self.matrix has been updated"""
        self.unravel_map(self.matrix)
        equal_arrays = np.all(self.matrix == np.zeros(self.dim))

        if equal_arrays:
            print("Error: no galaxies have been created!")


    def add_noise(self, noise):
        """ add normal-distributed noise to the dataset with strength noise"""
        if len(self.dim) == 3:
            normal = np.random.randn(self.dim[0],self.dim[1], self.dim[2])

        if len(self.dim) == 2:
            normal = np.random.randn(self.dim[0],self.dim[1])

        return noise*normal

    def generate_galaxy(self, noise=0):
        """ Generate a galaxy map"""
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

        #noise_ = noise*np.ones(self.dim)
        galaxy = galaxy + self.add_noise(noise)

        self.matrix += galaxy

        return galaxy


    def generate_DM(self, noise=0):
        """ Generate a DM map """
        dark_matter = np.zeros(self.dim)
        middle_row = int(self.dim[0]/2)
        middle_col = int(self.dim[1]/2)

        for i in range(self.dim[0]):
            for j in range(self.dim[1]):

                r = np.sqrt(   np.abs(middle_row-i)**2 + np.abs(middle_col-j)**2)

                for E in range(self.dim[2]):

                    dark_matter[i,j,E] = DM_profile(r)*DM_spectrum(E)

        dark_matter = dark_matter + self.add_noise(noise)

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
        """Sum up all energy slices"""
        data_ = self.unravel_map(data)
        data_combined = np.sum(data_, axis=2)

        return data_combined


    def display(self, data=None, slice=None, save_as=None, lim=None):
        """ Display the generated map, or a map from file. Set an upper limit
        on the colorbar with lim=max-value-of-colorbar for easy comparison with
        other map displays. To save set save_as=filename. """

        if data is not None:
            data = data
        else:
            self.check_matrix()
            data = self.matrix

        fig = plt.figure()

        if slice is not None:
            #plt.title("Energy slice: {:}".format(slice))
            data_ = self.unravel_map(data)
            data_ = data_[:,:,slice]
        else:
            #plt.title("Energy summed")
            data_ = self.combine_slices(data)

        #https://imagine.gsfc.nasa.gov/science/toolbox/gamma_generation.html
        y_axis = data_.shape[0]/2
        x_axis = data_.shape[1]/2
        axis_range = [-x_axis,x_axis,-y_axis, y_axis]

        plt.xlabel('Galactic Longitude')
        plt.ylabel('Galactic Latitude')

        ax = plt.gca()

        if lim is not None:
            im = ax.imshow(data_, cmap="inferno", extent = axis_range, vmax=lim)
        else:
            im = ax.imshow(data_, cmap="inferno", extent = axis_range)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

        if save_as: fig.savefig(save_as)


    def display_spectrum(self):

        fig1 = plt.figure()

        mat = self.matrix

        spectrum = np.sum(mat, axis = (0,1))

        plt.plot(spectrum)


#=============================================================================

def generate_data(nMaps, dim, noise = 0, PATH=None):
    """
    Generates nMaps of galaxies and dark matter, both with dimentions
    dim = (n,m,e) using the SkyMap class. The maps are raveled and stored as
    a row in a numpy array of dimentions (nMaps, n*m*e). If a path is specified,
    the arrays are stored in files with filenames: DM_dim_nMaps_.csv and
    galaxy_dim_nMaps_.csv
    ---------------
    Input:
        nMaps: int,   the number of maps to generate
        dim:   tuple, the chosen dimentions of the maps.
                      must either be (m,n) or (m,n,e) for energy
        noise: float, the strength of the noise
        PATH:  str,   the path to where the data should be stored
    ---------------
    Returns:
        galaxies:    ndarray, shape(nMaps, N), the galaxy maps
        dark_matter: ndarray, shape(nMaps, N), the DM maps
    """

    dim_ravel = np.prod(dim)    # dimentions of the raveled matrices
    galaxies = np.zeros((nMaps, dim_ravel))
    dark_matter = np.zeros((nMaps, dim_ravel))

    for i in range(nMaps):
        map = SkyMap(dim)

        galaxy = map.generate_galaxy(noise)
        dm = map.generate_DM(noise)

        galaxies[i,:] = map.ravel_map(galaxy)
        dark_matter[i,:] = map.ravel_map(dm)

    if PATH is not None:
        filename1 = "galaxy_{:}_{:}_".format(dim, 2*nMaps)
        filename2 = "DM_{:}_{:}_".format(dim, 2*nMaps)
        np.savetxt(PATH+filename1+".csv", galaxies, fmt="%.16f")
        np.savetxt(PATH+filename2+".csv", dark_matter, fmt="%.16f")

    return galaxies, dark_matter


def read_data(PATH, dim, n_maps_in_file, combine=True, ddf=False, shuf=True):
    """
    Reads the datafile created by the function generate_data and outputs
    the full dataset that contains all the galaxies with DM and all galaxies
    without DM. The first element of each row in the returned dataset is an
    indicator of wether the map in that row is with dark matter (1) or without
    dark matter(0). The dataset can either be returned as a Pandas dataframe or
    as a numpy array.
    ---------------
    Input:
        PATH:  str, path to where the data is stored
        dim:   tuple, shape (m,n,e) or (m,n), the dimentions of the dataset
        ddf:   bool, return the data as a pandas dataframe
        shuf:  bool, if you want the rows to be shuffled
        n_maps_in_file: int, the number of maps (last int in filename)
    ---------------
    returns:
        all: ndarray, shape(n_maps_in_file, prod(dim))
    """

    if n_maps_in_file < 2:
        print("Not possible with n_maps_in_file < 2.")
        sys.exit(1)

    filename1 = "galaxy_{:}_{:}_.csv".format(dim, n_maps_in_file)
    filename2 = "DM_{:}_{:}_.csv".format(dim, n_maps_in_file)

    galaxies = np.loadtxt(PATH+filename1)
    dark_matters = np.loadtxt(PATH+filename2)

    if n_maps_in_file == 2:
        galaxies = galaxies.reshape(1,-1)
        dark_matters = dark_matters.reshape(1,-1)

    # Create arrays with shape (n_maps_in_file, 1) with bool values
    trues = np.ones((dark_matters.shape[0],1), dtype=bool)
    falses = np.zeros((galaxies.shape[0],1), dtype=bool)

    if combine==True:
        # add the dark matter to the galaxies to create galaxies with DM
        dark_matters = galaxies+dark_matters

    # Add the bool value as the first array in the matrices containing the
    # maps. True appears as 1 and false appears as 0 in the stacked array.
    galaxies_ = np.hstack((falses, galaxies))
    dark_matters_ = np.hstack((trues, dark_matters))

    # Stack the full datasets on top of eachother
    all = np.vstack((galaxies_, dark_matters_))

    if shuf == True:
        # Shuffle the rows
        all = shuffle(all, random_state=42)

    if ddf ==True:
        col = ["Type"]+[i for i in range(galaxies.shape[1])]
        df = pd.DataFrame(all, columns=col)
        return df

    return all


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


def main_gert_new():
    """
    This one does the same as the old main_gert function did before changes
    """
    E = 4
    map = SkyMap(dim=(50,100,10))
    gal = map.generate_galaxy()[:,:,E]
    galn = gal + map.generate_galaxy_noise(noise=2)[:,:,E]
    dm = map.generate_DM()[:,:,E]
    dmn = dm + map.generate_DM_noise(noise=1)[:,:,E]
    fig, ax = plt.subplots(nrows=1, ncols=3,  figsize=(10, 3))
    ax[0].imshow(gal)
    ax[1].imshow(dm)
    ax[2].imshow(dmn+galn)
    map.display_spectrum()
    plt.show()


def main_alida():
    PATH="../data/"
    dim = (50,100,10)
    nm = 2

    # Method 1
    data = read_data(PATH, dim=dim, n_maps_in_file=nm, combine=False, shuf=False)

    gal_data = data[0][1:]
    dm_data = data[1][1:]
    comb_data = gal_data + dm_data

    map_data = SkyMap(dim=dim)
    map_data.set_matrix(comb_data)
    map_data.display(slice=5)
    map_data.display_spectrum()

    # method 2
    data1 = read_data(PATH, dim=dim, n_maps_in_file=nm, combine=True, shuf=False)

    gal_data1 = data1[0][1:]
    comb_data1 = data1[1][1:]

    map_data1 = SkyMap(dim=dim)
    map_data1.set_matrix(comb_data1)
    map_data1.display(slice=5)
    map_data1.display_spectrum()

    # method 3
    map = SkyMap(dim=(50,100,10))
    gal = map.generate_galaxy(noise=0.9)
    dm = map.generate_DM(noise=0.9)
    map.display(slice=5)
    map.display_spectrum()

    plt.show()

if __name__ == '__main__':
    #PATH = "../data/"
    #generate_data(nMaps=1, dim=(50,100,10), noise=0, PATH=PATH)
    #main_gert()
    #main_gert_new()
    main_alida()
