#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import quickplot as qupl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.utils import shuffle
import random
import sys


def gaussian_dot(i, j, max_val, sigma, dim, max_rad):

    mat = np.zeros(dim)

    for it in np.arange(int(max(0,i-3*sigma)),int(min(dim[0],i+3*sigma+1))):
        for jt in np.arange(int(max(0,j-3*sigma)),int(min(dim[1],j+3*sigma+1))):

            r = np.sqrt( (it-i)**2 + (jt-j)**2)

            mat[it,jt] = max_val*np.exp( -(r**2)/(2*sigma)  )

    return mat

class linear_planar_profile():
    def __init__(self, gradient = 5, max_val=50):
        self.gradient = gradient
        self.max_val = max_val
    def func(self,x):
        return self.max_val - self.gradient*x

class linear_spherical_profile():
    def __init__(self, gradient = 5, max_val=10):
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
    def __init__(self, sigma = 2, max_val=0.0, mean=10):
        self.sigma = sigma
        self.max_val=max_val
        self.mean = mean
    def func(self,E):
        return self.max_val*np.exp(- ( (E - self.mean)**2 )/(2*self.sigma**2) )

class linear_spectrum():
    def __init__(self, max_val=1, grad=5):
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
    """
    Class that generates a galactic center (GC) object either with or without a
    dark matter component.

    Parameters:
    -----------
    dim:  tuple, shape (h,w,e)
        The dimentions of the GC where h is the height, w is the width and e
        is the number of different energy levels.

    is_dm: bool, default=False
        Boolean to indicate if the GC object should or should not contain a
        dark matter component. By default the GC object is created without
        dark-matter.

    are_irreg: bool, default=False
        Boolean to indicate if you want to add static to the galaxy. The static
        is generated using random walk (TODO: elaborate?)

    noise_level: float, default=1
        The level of the Gaussian noise added to the GC object.

    Attributes:
    -----------
    matrix: ndarray, shape (h,w,e)
        Description of what this is

    matrix_galaxy: ndarray, shape (h,w,e)
        Description of what this is

    matrix_dm: ndarray, shape (h,w,e)
        Description of what this is


    Methods:
    --------
    display(): display an illustration of the galaxy. Optional arguments are,

        data:    You can also choose to only display any one of the
                 attributes whose name starts with matrix_, see example.

        slice:   Choose one energy level to look at, by default all
                 energy levels are summed when not specifying a slice.

        save_as: If you want to save the figure simply specify a filename
                 you want to save the figure as.

        lim:     Set a maximum limit on the colourbar (useful for comparing)


    spectrum(): display the cumulative counts


    Examples
    --------
    Generate a GC object with a DM component and display it

    >>> from generate import SkyMap
    >>> map = SkyMap(dim=(50,100,10), is_dm=True)
    >>> map.display()

    To see the counts per energy do

    >>> map.spectrum()

    You can also choose to display only the galaxy or DM by doing

    >>> map.display(map.matrix_galaxy)
    >>> map.display(map.matrix_dm)

    or any of the other attributes

    >>> map.display(map.matrix_galactic_center)
    >>> map.display(map.matrix_galactic_plane)
    >>> map.display(map.matrix_noise)
    >>> map.display(map.matrix_irregularities)

    """
    def __init__(self, dim, is_dm=False, dm_strength=1, are_irreg=True, noise_level=1):

        self.noise_level = noise_level
        self.dim = dim
        self.is_dm=is_dm
        self.are_irreg = are_irreg
        self.dm_strength = dm_strength

        # self.galactic_plane_max_profile = 100

        # self.galactic_plane_max_spectrum = 100

        # self.galactic_center_max_profile = 0

        # self.galactic_center_max_spectrum = 0

        # self.dm_max_profile = 10

        # self.dm_max_spectrum = 100


        self.galactic_plane_profile = linear_planar_profile(max_val  = 100,
                                                            gradient = 100/self.dim[0] )

        self.galactic_plane_spectrum = linear_spectrum(max_val  = 100,
                                                       grad = 100/self.dim[2] )

        self.galactic_center_profile = gaussian_spherical_profile(max_val=0)

        self.galactic_center_spectrum = linear_spectrum(max_val=0)

        self.dm_profile = linear_spherical_profile(max_val  = 100,
                                                   gradient = 100/self.dim[0])

        self.dm_spectrum = gaussian_spectrum(max_val = 10,
                                             sigma   = self.dim[2]/15,
                                             mean    = self.dim[2]/2 )



        self.noise = gaussian_noise(dim=self.dim, noise_level = noise_level)




        self.n_steps_frac = 0.5
        self.horizontal_step = 5
        self.vertical_step = 1
        self.takeover_step = 1
        self.takeover_margin = 0.01
        self.use_gaussian=False
        self.sigma_dot=0.2


        self.matrix = np.zeros(self.dim)

        self.matrix_galaxy = np.zeros(self.dim)

        self.matrix_galactic_center = np.zeros(self.dim)

        self.matrix_galactic_plane = np.zeros(self.dim)

        self.matrix_dm = np.zeros(self.dim)

        self.matrix_noise = np.zeros(self.dim)


        self.matrix_irregularities = np.zeros(self.dim)



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


    def generate_galaxy(self):
        """ Generate a galaxy map"""

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

        self.matrix_galaxy = galaxy

        if self.are_irreg: self.generate_irregularities(n_steps_frac=self.n_steps_frac, horizontal_step=self.horizontal_step,
                                                        vertical_step=self.vertical_step, takeover_step=self.takeover_step,
                                                        takeover_margin=self.takeover_margin, sigma_dot=self.sigma_dot,
                                                        use_gaussian=self.use_gaussian)

        self.matrix_galactic_plane = galactic_plane

        self.matrix_galactic_center = galactic_center

        # self.matrix_galaxy = galaxy

        self.matrix = self.matrix_galaxy + self.matrix_dm + self.matrix_noise

        return galaxy


    def generate_irregularities(self, n_steps_frac=0.5, horizontal_step=2, vertical_step=1, takeover_step=1, takeover_margin=0.01, sigma_dot=0.2, use_gaussian=False):

        galaxy = self.matrix_galaxy

        sig = max(horizontal_step*self.dim[1]/100,1)
        takeover_step = max(takeover_step*self.dim[1]/100,1)

        for E in range(self.dim[2]):

            pos_i = int(self.dim[0]/2)
            pos_j = int(self.dim[1]/2)


            for step in range(int(self.dim[0]*self.dim[1]*n_steps_frac)):


                dis = int(self.dim[0]/2)-pos_i

                step_i = int(np.round(np.random.randn(1)[0]*(np.abs(dis)*vertical_step +1)))
                step_j = int(np.round(np.random.randn(1)[0]*sig))
                pos_i = (pos_i + step_i)%self.dim[0]
                pos_j = (pos_j + step_j)%self.dim[1]


                factor = takeover_margin*np.random.randn(1)[0] + 1

                take_i = pos_i + np.random.randint(-takeover_step,takeover_step+1)
                take_j = pos_j + np.random.randint(-takeover_step,takeover_step+1)

                take_i = (take_i)%self.dim[0]
                take_j = (take_j)%self.dim[1]

                extra = galaxy[take_i, take_j, E]*factor - galaxy[pos_i, pos_j, E]

                if use_gaussian: galaxy[:,:,E] += gaussian_dot(pos_i, pos_j, extra, sigma_dot, (self.dim[0],self.dim[1]), 0)
                else: galaxy[pos_i, pos_j, E] += extra



        self.matrix_irregularities = galaxy-self.matrix_galaxy
        self.matrix_galaxy = galaxy



    def generate_dm(self):
        """ Generate a DM map """
        dark_matter = np.zeros(self.dim)
        middle_row = int(self.dim[0]/2)
        middle_col = int(self.dim[1]/2)

        for i in range(self.dim[0]):
            for j in range(self.dim[1]):

                r = np.sqrt(np.abs(middle_row-i)**2 + np.abs(middle_col-j)**2)

                for E in range(self.dim[2]):

                    dark_matter[i,j,E] = self.dm_profile.func(r)*self.dm_spectrum.func(E)

        dark_matter = dark_matter*self.dm_strength

        self.matrix_dm = dark_matter

        self.matrix = self.matrix_galaxy + self.matrix_noise + dark_matter

        self.is_dm = True

        return dark_matter



    def generate_noise(self):
        """ Generate noise """
        self.matrix_noise = self.noise.func()
        self.matrix = self.matrix_galaxy + self.matrix_dm + self.matrix_noise


    def ravel_map(self, matrix):
        """ ravel the matrix """
        a = matrix.copy()
        b = a.ravel()
        self.dim_ravel= b.shape
        return b

    def unravel_map(self, data):
        """ restore original dimentions """
        a = data.copy()
        if len(a.shape)==1:
            d = a.reshape(self.dim)
        else:
            d = a

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
            data_ = data
        else:
            self.check_matrix()
            data_ = self.matrix

        fig = plt.figure()

        if slice is not None:
            data_ = self.unravel_map(data_)
            data_ = data_[:,:,slice]

        else:
            data_ = self.combine_slices(data_)

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

        #if data is None and self.is_dm==True:
        #    self.spectrum()

        plt.show()


    def spectrum(self):
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

    data_(2*nMaps, m, n, e)_{dm_strength}_{noise_level}_{random_walk}_.npy

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
        if (2*i % 100==0):
            print("Generating maps...{:.0f}/{:.0f}".format(2*(i+1), 2*nMaps))

        map_g = SkyMap(dim=dim, is_dm=False, are_irreg=random_walk, noise_level=noise_level)
        galaxy = map_g.matrix

        map_dm = SkyMap(dim=dim, is_dm=True, dm_strength=dm_strength,
                        are_irreg=random_walk, noise_level=noise_level)
        dm = map_dm.matrix

        galaxies[i,:] = map_g.ravel_map(galaxy)
        dark_matters[i,:] = map_dm.ravel_map(dm)

    # Create arrays with shape (n_maps_in_file, 1) with bool values and
    # add the bool value as the first array in the matrices containing the
    # maps. True appears as 1 and false appears as 0 in the stacked array.

    trues = np.ones((dark_matters.shape[0], 1), dtype=bool)
    falses = np.zeros((galaxies.shape[0], 1), dtype=bool)

    print(trues.shape)
    print(galaxies.shape)
    galaxies_ = np.hstack((falses, galaxies))
    dark_matters_ = np.hstack((trues, dark_matters))

    # Stack the full datasets on top of eachother
    all = np.vstack((galaxies_, dark_matters_))

    if shuf == True:
        all = shuffle(all, random_state=42)

    if PATH is not None:
        tuple = (2*nMaps, dim[0], dim[1], dim[2])
        fn = "data_{:}_{:}_{:}_{:}_".format(tuple, dm_strength, noise_level, random_walk)
        #np.savetxt(PATH+fn+".csv", all, fmt="%.16f")
        np.save(PATH+fn, all)

    return all

def generate_data_v2(nMaps, dim, noise_level = 0, random_walk = True, shuf=True, PATH=None):
    """
    Generates nMaps of galaxies with random levels of dark matter strength
    using the SkyMap class. The maps are raveled and stored as a row in a
    numpy array where the first number in each row corresponds to the strength
    of the dark matter (to be used as labels). If a path is specified, the
    arrays are stored in a file with filename:

    maps_(nMaps, m, n, e)_{dm_strength}_{noise_level}_{random_walk}_.npy

    ---------------
    Input:
        nMaps:       int,   the number of maps to generate
        dim:         tuple, the chosen dimentions of the maps.(m,n,e)
        noise_level: float, the strength of the noise
        random_walk: bool,  if the data is generated with random walk
        shuf:        bool,  shuffle the rows of the data
        PATH:        str,   the path to where the data should be stored
    ---------------
    Returns:
        data: ndarray, shape (nMaps, n, m, e)
    """
    dim_ravel = np.prod(dim)    # dimentions of the raveled matrices
    dark_matters = np.zeros((nMaps, dim_ravel))
    dm_str = np.zeros(nMaps)

    for i in range(nMaps):
        if (i % 100==0):
            print("Generating maps...{:.0f}/{:.0f}".format((i+1), nMaps))

        dm_strength = random.random()

        map = SkyMap(dim=dim, is_dm=True, dm_strength=dm_strength,
                        are_irreg=random_walk, noise_level=noise_level)

        dm = map.matrix
        dark_matters[i,:] = map.ravel_map(dm)
        dm_str[i] = dm_strength

    labels = dm_str.reshape(-1,1)
    all = np.hstack((labels, dark_matters))

    if shuf == True:
        all = shuffle(all, random_state=42)

    if PATH is not None:
        tuple = (nMaps, dim[0], dim[1], dim[2])
        fn = "maps_{:}_{:}_{:}_".format(tuple, noise_level, random_walk)
        np.save(PATH+fn, all)

    return all


def load_data(file="", slice = None):
    """
    Reads the datafile created by the function generate_data
    ---------------
    Input:
        file:  str, filename
        slice: int, if you only want to see one energy bin
    ---------------
    returns:
        maps, labels, stats
    """
    info = file.split("_")[1:-1]
    info = [eval(elm) for elm in info]

    if len(info) == 3:
        keys = ["ndim", "noise", "walk"]

    if len(info) == 4:
        keys = ["ndim", "dm_strength", "noise", "walk"]

    # Create dictionary with information from filename
    stats = {keys[i]: info[i] for i in range(len(keys))}

    # Load the array from file
    data = np.load(file)

    labels = data[:,0].reshape(-1,1)
    maps = data[:,1:]

    # reshape the maps
    maps = maps.reshape(stats["ndim"])

    if slice is not None:
        ndim = stats["ndim"]
        ndim_new = ndim[:-1] + tuple([1])
        maps = maps[:,:,:, slice].reshape(ndim_new)

    return maps, labels, stats

def arguments():
    """ Generate data from command-line """

    import argparse
    description =  """Generate Galactic Center Excess pseudodata TBA"""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-n', type=int, metavar='--number_of_maps', action='store', default=1000,
                    help='The number of maps to generate for each type, default=1000')
    parser.add_argument('-d', type=str, metavar='--dimentions', action='store', default="28,28,10",
                    help="Dimentions of the maps use as: -d dim1,dim2,dim3, default=28,28,10")
    parser.add_argument('-dm', type=float, metavar='--dm_strength', action='store', default=1,
                    help='Only relevant when using -v 1: Strength of dark matter, default=1')
    parser.add_argument('-nl', type=float, metavar='--noise_level', action='store', default=1,
                    help='Level of gaussian nose in data, default=1')
    parser.add_argument('-r', type=str, metavar='--random_walk', action='store', default="True",
                    help='Use random walk, default=True')
    parser.add_argument('-s', type=str, metavar='--shuffle_maps', action='store', default="True",
                    help='Shuffle the maps before storing, default=True')
    parser.add_argument('-p', type=str, metavar='--PATH', action='store', default="../data/",
                        help='Path to where the data should be stored, default="../data/"')
    parser.add_argument('-v', type=int, metavar='--version', action='store', default=1,
                    help='Choose the version of generator: 1 for v1 or 2 for v2, default=1')
    args = parser.parse_args(["-h"])


    if args.v != 1 or args.v != 2:
        error = "You need to choose a valid version number:\nversion 1: -v 1\nversion 2: -v 2"
        parser.error(error)


    n, d, dm, nl = args.n, eval(args.d), args.dm, args.nl,
    r, s, p, v = eval(args.r), eval(args.s), args.p, args.v

    return n, d, dm, nl, r, s, p, v


if __name__ == "__main__":
    from datetime import datetime
    start_time = datetime.now()

    n, d, dm, nl, r, s, p, v = arguments()

    if v == 1:
        generate_data(nMaps=n, dim=d, noise_level=nl, random_walk=r, shuf=s, PATH=p)

    if v == 2:
        generate_data_v2(nMaps=n, dim=d, dm_strength=dm, noise_level=nl, random_walk=r, shuf=s, PATH=p)


    time_elapsed = datetime.now() - start_time
    print('\nTime elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
