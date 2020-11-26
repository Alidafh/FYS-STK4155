#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import generate as dat


def plots():
    PATH="../data/"
    FIG ="../figures/"
    dim = (50,100,10)
    nm=2

    data = dat.read_data(PATH, dim=dim, n_maps_in_file=nm, combine=False, shuf=False)

    gal = data[0][1:]
    dm = data[1][1:]
    comb = gal + dm

    map_data = dat.SkyMap(dim=dim)
    map_data.set_matrix(comb)
    map_data.display(gal, slice=5, lim=50, save_as=FIG+"galaxy.png")
    map_data.display(dm, slice=5, lim=50, save_as=FIG+"dm.png")
    map_data.display(slice=5, lim=50, save_as=FIG+"combined.png")

    plt.show()

plots()

"""
def plot_illustrations(dim, E):
    PATH="../data/"

    data = gen.read_data(PATH, dim=dim, shuf=False, ddf = False, combine=False)

    g = data[0][1:].reshape(dim)
    d = data[7][1:].reshape(dim)
    c = g+d

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3,  figsize=(20, 3), sharex=True)
    y_axis = dim[0]/2
    x_axis = dim[1]/2
    axis_range = [-x_axis,x_axis,-y_axis, y_axis]

    im0 = ax0.imshow(g[:,:,E], cmap="inferno", vmax=50, extent = axis_range)
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im0, cax=cax0)

    im1 = ax1.imshow(d[:,:,E], cmap="inferno",vmax=50,extent = axis_range)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax1)

    im2 = ax2.imshow(c[:,:,E], cmap="inferno",vmax=50,extent = axis_range)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, cax=cax2)
    plt.show()

plot_illustrations(dim = (50, 100, 10), E=5)
"""
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


def combine_and_shuffle1(PATH, filename_g, filename_dm):

    #galaxy, dim_galaxy = read_data(PATH, filename_g)
    #dark_matter, dim_dm = read_data(PATH, filename_dm)

    #if dim_galaxy != dim_dm:
    #    print("Mis-match between dimentions of DM and Galaxy datasets")
    #    sys.exit(1)

    dim = (2,2,2)
    galaxy, dark_matter = generate_data(5, dim=dim)

    keys = ["x{:}".format(i) for i in range(np.prod(dim))]
    data = {}
    for j in range(galaxy.shape[0], 2*dark_matter.shape[0]):
        for i in range(galaxy.shape[0]):
            tmp_g = {}
            tmp_d = {}
            tmp_g["bool"] = False
            tmp_d["bool"] = True
            for key, g, d in zip(keys, galaxy[i], dark_matter[i]):
                tmp_g[key] = g
                tmp_d[key] = d

            data[i] = tmp_g
            data[j] = tmp_d

        print("Galaxy ", np.array2string(galaxy[i], formatter={'float_kind':'{0:.6f}'.format}))
        print("DM     ", np.array2string(dark_matter[i], formatter={'float_kind':'{0:.6f}'.format}))


    df = pd.DataFrame.from_dict(data, orient='index')
    print(df)

"""

"""
def make_dataframe(all, galaxies, dark_matters, shuf=True):

    Creates a Pandas DataFrame contining both the galaxy datasets and the
    dark matter datasets. The first column in the dataframe indicates wether
    the dataset contains a galaxy with or without dark matter:
        with DM:    Type = 1
        without DM: Type = 0
    ---------------
    Input
        galaxies:      ndarray (nMaps, N), galaxy maps
        dark_matters:  ndarray (nMaps, N), dark matter maps
    ---------------
    Output
        df: pandas dataframe

    # Create arrays with shape (nMaps, 1) with bool values
    trues = np.ones((dark_matters.shape[0],1), dtype=bool)
    falses = np.zeros((galaxies.shape[0],1), dtype=bool)

    # add the dark matter to the galaxies
    dark_matters = galaxies+dark_matters

    # Add the bool value as the first array in the matrices containing the
    # maps. True appears as 1 and false appears as 0 in the stacked array.
    galaxies_ = np.hstack((falses, galaxies))
    dark_matters_ = np.hstack((trues, dark_matters))

    # Stack the full datasets on top of eachother and make a pd.DataFrame
    all= np.vstack((galaxies_, dark_matters_))

    col = ["Type"]+[i for i in range(galaxies.shape[1])]
    df = pd.DataFrame(all, columns=col)

    # Shuffle the rows of the dataframe
    if shuf == True:
        df = shuffle(df, random_state=42)
        df.reset_index(inplace=True, drop=True)

    return df
"""
