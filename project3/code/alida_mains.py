#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import generate as gen


def main0():
    PATH="../data/"
    gen.generate_data(nMaps=5, dim=(50,100,10), PATH=PATH)

def main1():
    PATH="../data/"
    galaxy, dim = gen.read_data(PATH, filename="galaxy_(50, 100, 10)_.csv")
    dm, dim = gen.read_data(PATH, filename="DM_(50, 100, 10)_.csv")
    combined = galaxy+dm

    map1 = gen.SkyMap(dim)
    map1.display(dm[0], slice=5, save_as="../figures/test_DM.png", lim=50)
    plt.show()

    map1.display(galaxy[0], slice=5, save_as="../figures/test_Gal.png", lim=50)
    plt.show()

    map1.display(combined[0], slice=5, save_as="../figures/test_combo.png", lim=50)
    plt.show()

def main2():
    PATH="../data/"
    galaxy, dim = gen.read_data(PATH, filename="galaxy_(50, 100, 10)_.csv")
    dm, dim = gen.read_data(PATH, filename="DM_(50, 100, 10)_.csv")
    combined = galaxy+dm
    g = galaxy[0].reshape(dim)
    d = dm[0].reshape(dim)
    c = combined[0].reshape(dim)

    E = 5
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3,  figsize=(20, 3))
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

#main0()
main1()
#main2()



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
