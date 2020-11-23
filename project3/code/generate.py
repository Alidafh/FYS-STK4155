#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return 2*x

def DM_profile(r):
    # return 1/(r+0.1)
    return 10 - r/2.5


def DM_spectrum(E):
    
    sigma = 1
    mean = 5
    norm = 10
    
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
        return galaxy



    def generate_DM(self, noise):
        dark_matter = np.zeros(self.dim)
        middle_row = int(self.dim[0]/2)
        middle_col = int(self.dim[1]/2)

        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                
                r = np.sqrt(   np.abs(middle_row-i)**2 + np.abs(middle_col-j)**2     )
                
                for E in range(self.dim[2]):
                
                    dark_matter[i,j,E] = DM_profile(r)*DM_spectrum(E)


        return dark_matter
    
    
    
    
    
    
    def generate_galaxy_noise(self, noise):
        
        noise = noise*np.random.randn(self.dim[0],self.dim[1], self.dim[2])
        
        return noise
    
    
    
    
    
    def generate_DM_noise(self, noise):
        
        noise = noise*np.random.randn(self.dim[0],self.dim[1], self.dim[2])
        
        return noise




    def write_to_file(self):
        return 0

    def read_from_file(self):
        return 0

    def ravel_map(self, matrix):
        b = matrix.ravel()
        return b

    def unravel_map(self, matrix_raveled):
        d = matrix_raveled.reshape(self.dim)
        return d

    def Display(self):
        return 0


def main():
    
    E = 8
    map = SkyMap(dim=(50,100,10))
    gal = map.generate_galaxy(noise=0.1)[:,:,E]
    galn = gal + map.generate_galaxy_noise(noise=2)[:,:,E]
    dm = map.generate_DM(noise=0.1)[:,:,E]
    dmn = dm + map.generate_DM_noise(noise=1)[:,:,E]
    fig, ax = plt.subplots(nrows=1, ncols=3,  figsize=(10, 3))
    ax[0].imshow(gal)
    ax[1].imshow(dm)
    ax[2].imshow(dmn+galn)
    plt.show()

if __name__ == '__main__':
    main()












































