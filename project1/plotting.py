#!/usr/bin/python
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_3D(x, y, z, title, filename):
    """
    makes a 3d plot and saves it in the folder: output/figures/
    --------------------------------
    Input
        x,y,z:
        title: title of the figure
        filename: the name of the saved file
    --------------------------------
    TODO: change back to saving in PDF format
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x,y,z,cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.title(title)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    #plt.savefig("output/figures/{}.pdf".format(filename), scale=0.1)
    plt.savefig("output/figures/{}.png".format(filename), scale=0.1)
    print("     Figure saved in: output/figures/{}.pdf\n".format(filename))
