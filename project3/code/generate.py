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

    def generate_galaxy(self, noise):
        galaxy = np.zeros(self.dim)
        middle_row = int(self.dim[0]/2)

        for i in range(self.dim[0]):    # loop over rows
            if i == middle_row:
                galaxy[middle_row,:] = func(middle_row)*np.ones(self.dim[1])
            if i < middle_row:
                galaxy[i,:] = func(i)*np.ones(self.dim[1])
            if i > middle_row:
                x = np.abs(i-2*middle_row)
                galaxy[i,:] = func(x)*np.ones(self.dim[1])

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
                    print("MIDDLEE")

        return dark_matter

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
    map = SkyMap(dim=(50,100))
    gal = map.generate_galaxy(noise=0.1)
    dm = map.generate_DM(noise=0.1)
    fig, ax = plt.subplots(nrows=1, ncols=3,  figsize=(10, 3))
    ax[0].imshow(gal)
    ax[1].imshow(dm)
    ax[2].imshow(gal+dm)
    plt.show()

if __name__ == '__main__':
    main()
