#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import quickplot as qupl
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
    
    
    def __init__(self, dim, is_dm=False):
        
        
        self.noise_level = 1
        
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


     
        
        

    def generate_galaxy(self):
        
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
                
                r = np.sqrt(   np.abs(middle_row-i)**2 + np.abs(middle_col-j)**2     )
                
                for E in range(self.dim[2]):
        
                    galactic_center[i,j,E] = self.galactic_center_profile.func(r)*self.galactic_center_spectrum.func(E)
        
        galaxy = galactic_plane + galactic_center
        
        
        pos_i = middle_row
        pos_j = middle_col
        
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
    
    map1 = SkyMap(dim=(50,100,10), is_dm = True)
    
    
    
    sky = map1.matrix[:,:,E]
    
    
    gal = map1.matrix_galaxy[:,:,E]
    
    dm = map1.matrix_dm[:,:,E]
    

    fig, ax = plt.subplots(nrows=1, ncols=3,  figsize=(10, 3))
    ax[0].imshow(gal)
    ax[1].imshow(dm)
    ax[2].imshow(sky)
    map1.display_spectrum()
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


# if __name__ == '__main__':
#     #main0()
#     main_gert()


E = 6
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

