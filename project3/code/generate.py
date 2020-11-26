#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import quickplot as qupl
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.utils import shuffle
import sys



class linear_planar_profile():
    def __init__(self, gradient = 2, max_val=50):
        self.gradient = gradient
        self.max_val = max_val
    def func(self,x):
        return self.max_val - self.gradient*x

class linear_spherical_profile():
    def __init__(self, gradient = 1/2.5, max_val=10):
        self.gradient = gradient
        self.max_val=max_val
    def func(self,x):
        return max(  self.max_val - x*self.gradient  , 0 )


class gaussian_spherical_profile():
    def __init__(self, sigma = 10, max_val=1, mean=0):
        self.sigma = sigma
        self.max_val=max_val
        self.mean = mean
    def func(self,x):
        return self.max_val*np.exp(- ( (x - self.mean)**2 )/(2*self.sigma**2) )



class gaussian_spectrum():
    def __init__(self, sigma = 5, max_val=2, mean=50):
        self.sigma = sigma
        self.max_val=max_val
        self.mean = mean
    def func(self,E):
        return self.max_val*np.exp(- ( (E - self.mean)**2 )/(2*self.sigma**2) )


class linear_spectrum():
    def __init__(self, max_val=1, grad=0.1):
        self.max_val=max_val
        self.grad = grad
    def func(self,E):
        return self.max_val - self.grad*E


class gaussian_noise():
    def __init__(self, noise_level=1, dim=(50,100,10)):
        self.noise_level = noise_level
        self.dim = dim
    def func(self):
        return self.noise_level*np.random.randn(self.dim[0],self.dim[1], self.dim[2])

class SkyMap:


    def __init__(self, dim, is_dm=False, noise_level=1):
        # Add DM normalization constant to input/self.

        self.noise_level = noise_level

        self.galactic_plane_max_profile = 50

        self.galactic_plane_max_spectrum = 100

        self.galactic_center_max_profile = 10

        self.galactic_center_max_spectrum = 100

        self.dm_max_profile = 25

        self.dm_max_spectrum = 10


        self.dim = dim

        self.matrix = np.zeros(self.dim)

        self.matrix_galaxy = np.zeros(self.dim)

        self.matrix_galactic_center = np.zeros(self.dim)

        self.matrix_galactic_plane = np.zeros(self.dim)

        self.matrix_dm = np.zeros(self.dim)

        self.matrix_noise = np.zeros(self.dim)


        self.galactic_plane_profile = linear_planar_profile(max_val = self.galactic_plane_max_profile)

        self.galactic_plane_spectrum = linear_spectrum(max_val = self.galactic_plane_max_spectrum)

        self.galactic_center_profile = gaussian_spherical_profile(max_val=self.galactic_center_max_profile)

        self.galactic_center_spectrum = linear_spectrum(max_val=self.galactic_center_max_spectrum)

        self.dm_profile = linear_spherical_profile(max_val = self.dm_max_profile)

        self.dm_spectrum = gaussian_spectrum(max_val = self.dm_max_spectrum)

        self.noise = gaussian_noise(dim=self.dim, noise_level=self.noise_level)


        self.is_dm=is_dm


        self.generate_galaxy()

        self.generate_noise()

        if is_dm: self.generate_dm()


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


    def generate_galaxy(self):
        """ Generate a galaxy map """
        galactic_plane = np.zeros(self.dim)
        galactic_center = np.zeros(self.dim)

        middle_row = int(self.dim[0]/2)
        middle_col = int(self.dim[1]/2)

        for i in range(self.dim[0]):    # loop over rows

            d = np.abs(i - middle_row)

            for E in range(self.dim[2]):

                galactic_plane[i,:,E] = self.galactic_plane_profile.func(d)*np.ones(self.dim[1])*self.galactic_plane_spectrum.func(E)


        for i in range(self.dim[0]):
            for j in range(self.dim[1]):

                r = np.sqrt(np.abs(middle_row-i)**2 + np.abs(middle_col-j)**2)

                for E in range(self.dim[2]):

                    galactic_center[i,j,E] = self.galactic_center_profile.func(r)*self.galactic_center_spectrum.func(E)

        galaxy = galactic_plane + galactic_center


        #pos_i = middle_row
        #pos_j = middle_col

        self.walk = np.zeros(self.dim)

        for E in range(self.dim[2]):

            pos_i = middle_row
            pos_j = middle_col

            up = 0
            down = 0

            for step in range(int(self.dim[0]*self.dim[1]*0.5)):

                sig = 2
                dis = middle_row-pos_i
                # step_i = np.random.randint(min(-5,dis),max(5,dis))
                # step_j = np.random.randint(-2,2)
                step_i = int(np.round(np.random.randn(1)[0]*(np.abs(dis) +1)))
                step_j = int(np.round(np.random.randn(1)[0]*sig))
                pos_i = (pos_i + step_i)%self.dim[0]
                pos_j = (pos_j + step_j)%self.dim[1]

                # if pos_i >= self.dim[0]-3 or pos_i <= 3: pos_i -= 2*step_i
                # if pos_j >= self.dim[1]-3 or pos_j <= 3: pos_j -= 2*step_j

                factor = 0.01*np.random.randn(1)[0] + 1

                take_i = pos_i + np.random.randint(-1,1)
                take_j = pos_j + np.random.randint(-1,1)

                take_i = (take_i)%self.dim[0]
                take_j = (take_j)%self.dim[1]

                before = galaxy[pos_i, pos_j, E]

                galaxy[pos_i, pos_j, E] = galaxy[take_i, take_j, E]*factor

                after = galaxy[pos_i, pos_j, E]


                # if before < after and pos_i > middle_row: up += 1
                # if before > after and pos_i > middle_row: down += 1

                # self.walk[pos_i,pos_j,E] += 1

            # print(E, up, down)


        self.matrix_galactic_plane = galactic_plane

        self.matrix_galactic_center = galactic_center

        self.matrix_galaxy = galaxy

        self.matrix = galaxy + self.matrix_dm + self.matrix_noise

        return galaxy



    def generate_dm(self):
        """ Generate a DM map """
        dark_matter = np.zeros(self.dim)
        middle_row = int(self.dim[0]/2)
        middle_col = int(self.dim[1]/2)

        for i in range(self.dim[0]):
            for j in range(self.dim[1]):

                r = np.sqrt(   np.abs(middle_row-i)**2 + np.abs(middle_col-j)**2)

                for E in range(self.dim[2]):


                    dark_matter[i,j,E] = self.dm_profile.func(r)*self.dm_spectrum.func(E)


        self.matrix_dm = dark_matter

        self.matrix = self.matrix_galaxy + self.matrix_noise + dark_matter

        self.is_dm = True

        return dark_matter



    def generate_noise(self):

        self.matrix_noise = self.noise.func()
        self.matrix = self.matrix_galaxy + self.matrix_dm + self.matrix_noise


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
        """ cumulative count vs energy spectrum """
        # save figure or add to display()
        spectrum = np.sum(self.matrix, axis = (0,1))
        spectrum_dm = np.sum(self.matrix_dm, axis = (0,1))
        spectrum_galactic_plane = np.sum(self.matrix_galactic_plane, axis = (0,1))
        spectrum_galactic_center = np.sum(self.matrix_galactic_center, axis = (0,1))
        spectrum_galaxy = np.sum(self.matrix_galaxy, axis = (0,1))

        qp = qupl.QuickPlot()
        qp.reset()

        qp.grid = False
        # qp.y_log = True


        qp.plot_title = "Cumulative spectrum of the skymap"
        qp.x_label = "Energy (given by slice index)"
        qp.y_label = "Cumulative counts"


        qp.add_plot(np.arange(len(spectrum)), spectrum, 'k', "Total spectrum")
        qp.add_plot(np.arange(len(spectrum)), spectrum_dm, 'b--', "DM spectrum")
        qp.add_plot(np.arange(len(spectrum)), spectrum_galactic_plane, 'y:', "Galactic plane spectrum")
        qp.add_plot(np.arange(len(spectrum)), spectrum_galactic_center, 'm:', "Galactic center spectrum")
        qp.add_plot(np.arange(len(spectrum)), spectrum_galaxy, 'r--', "Total galactic spectrum")

        qp.create_plot("spectrum")


#=============================================================================

def generate_data(nMaps, dim, dm_strength=1, noise_level = 0, random_walk = True, shuf=True, PATH=None):
    """
    Generates nMaps of galaxies with dark matter and nMaps of galaxies without
    dark matter using the SkyMap class. The maps are raveled and stored as
    a row in a numpy array. If a path is specified, the arrays are stored in
    a file with filename:

    data_(2*nMaps, m, n, e)_{dm_strength}_{noise_level}_{random_walk}_.csv

    ---------------
    Input:
        nMaps:       int,   the number of maps to generate
        dim:         tuple, the chosen dimentions of the maps.(m,n,e)
        dm_strength: float, the DM normalization
        noise_level: float, the strength of the noise
        random_walk: bool,  if the data is generated with random walk
        shuf:        bool,  shuffle the rows of the data
        PATH:        str,   the path to where the data should be stored
    ---------------
    Returns:
        data: ndarray, shape (2*nMaps, n, m, e)
    """

    dim_ravel = np.prod(dim)    # dimentions of the raveled matrices
    galaxies = np.zeros((nMaps, dim_ravel))
    dark_matters = np.zeros((nMaps, dim_ravel))

    for i in range(nMaps):
        print("Generating maps...{:.0f}/{:.0f}".format(2*(i+1), 2*nMaps))
        map_g = SkyMap(dim=dim, noise_level=noise_level, is_dm=False)
        galaxy = map_g.matrix

        map_dm = SkyMap(dim=dim, noise_level=noise_level, is_dm=True)
        dm = map_dm.matrix

        galaxies[i,:] = map_g.ravel_map(galaxy)
        dark_matters[i,:] = map_dm.ravel_map(dm)

    # Create arrays with shape (n_maps_in_file, 1) with bool values and
    # add the bool value as the first array in the matrices containing the
    # maps. True appears as 1 and false appears as 0 in the stacked array.

    trues = np.ones((dark_matters.shape[0], 1), dtype=bool)
    falses = np.zeros((galaxies.shape[0], 1), dtype=bool)

    galaxies_ = np.hstack((falses, galaxies))
    dark_matters_ = np.hstack((trues, dark_matters))

    # Stack the full datasets on top of eachother
    all = np.vstack((galaxies_, dark_matters_))

    if shuf == True:
        all = shuffle(all, random_state=42)

    if PATH is not None:
        tuple = (2*nMaps, dim[0], dim[1], dim[2])
        fn = "data_{:}_{:}_{:}_{:}_".format(tuple, dm_strength, noise_level, random_walk)
        np.savetxt(PATH+fn+".csv", all, fmt="%.16f")

    return all


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

def load_data(PATH="../data/", file="data_(40, 100, 50, 10)_1_0_True_.csv", slice = None):
    """
    Not done
    """
    # Create dictionary with information from filename
    keys = ["ndim", "dm_strength", "noise", "walk"]

    info = file.split("_")[1:-1]
    info = [eval(elm) for elm in info]

    stats = {keys[i]: info[i] for i in range(len(keys))}

    # Load the array from file
    data = np.loadtxt(PATH+file)

    labels = data[:,0]
    maps = data[:,1:]

    # reshape the maps
    maps = maps.reshape(stats["ndim"])

    if slice is not None:
        ndim = stats["ndim"]
        ndim_new = ndim[:-1] + tuple([1])
        maps = maps[:,:,:, slice].reshape(ndim_new)

        #stats["input_shape"] = ndim_new[1:]


    return maps, labels, stats


def main_gert():
    E = 50
    map1 = SkyMap(dim=(50,100,100), is_dm = True)

    sky = map1.matrix[:,:,E]

    gal = map1.matrix_galaxy[:,:,E]

    dm = map1.matrix_dm[:,:,E]

    fig, ax = plt.subplots(nrows=1, ncols=3,  figsize=(10, 3))
    ax[0].imshow(gal)
    ax[1].imshow(dm)
    ax[2].imshow(sky)
    map1.display_spectrum()
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
    gal = map.generate_galaxy()
    dm = map.generate_dm()
    map.display(slice=5)
    map.display_spectrum()

    plt.show()


if __name__ == '__main__':
    from datetime import datetime
    start_time = datetime.now()

    PATH = "../data/"
    dim = (50,100,10)
    generate_data(nMaps=50, dim=dim, dm_strength=1, noise_level=0, random_walk=True, shuf=True, PATH=PATH)
    #data = load_data()
    time_elapsed = datetime.now() - start_time
    print('\nTime elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))



"""
    train_size = 0.8
    test_size = 1 - train_size
    X_train, X_test, y_train, y_test = train_test_split(maps, labels, train_size=train_size, test_size=test_size)

    return (X_train, y_train), (X_test, y_test)
"""
