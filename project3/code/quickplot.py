#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:09:23 2020

@author: gert
"""

import numpy as np
import matplotlib.pyplot as plt





class QuickPlot:


    def __init__(self):

        self.label_size = 30                     # Size of title and x- and y- labels

        self.line_width = 3                      # Thickness of plotted lines

        self.label_pad = 17                      # Distance between axis label and axis

        self.tick_size = 22                      # Size of axis ticks

        self.legend_size = 22                    # Size of legend

        self.grid = True                         # Determines whether the plot has a grid

        self.plot_title = None                   # Title of plot

        self.x_label = None                      # Label on x-axis

        self.y_label = None                      # Label on y-axis

        self.x_log = False                       # Determines whether the x-axis should have a logarithmic scale

        self.y_log = False                       # Determines whether the y-axis should have a logarithmic scale

        self.y_max = None                        # Determines the maximum y value to be shown on the plot

        self.y_min = None                        # Determines the minimum y value to be shown on the plot

        self.x_max = None                        # Determines the maximum x value to be shown on the plot

        self.x_min = None                        # Determines the minimum x value to be shown on the plot

        self.plots = []


    def create_plot(self, fig_label):

        plt.figure(fig_label)

        for plot in self.plots:
            plt.plot(plot.x, plot.y, plot.symbol, label=plot.label, linewidth=plot.linewidth)

        plt.title(self.plot_title,size=self.label_size)
        plt.xlabel(self.x_label,size=self.label_size, labelpad=self.label_pad)
        plt.ylabel(self.y_label,size=self.label_size, labelpad=self.label_pad)

        if self.x_log: plt.xscale("log")
        if self.y_log: plt.yscale("log")

        if self.y_max: plt.ylim(top=self.y_max)
        if self.y_min: plt.ylim(bottom=self.y_min)
        if self.x_max: plt.ylim(top=self.x_max)
        if self.x_min: plt.ylim(bottom=self.x_min)

        plt.legend(fontsize=self.legend_size)
        plt.xticks(fontsize=self.tick_size)
        plt.yticks(fontsize=self.tick_size)

        plt.grid(self.grid, which='both', axis='both')

        plt.show()

    def add_plot(self, x, y, symbol, label):

        self.plots.append(OnePlot(x, y, symbol, label, linewidth = self.line_width))


    def reset(self):

        self.plots = []

        self.plot_title = None

        self.x_label = None

        self.y_label = None

class OnePlot:

    def __init__(self, x, y, symbol, label, linewidth=3):

        self.x = x
        self.y = y
        self.symbol = symbol
        self.label = label
        self.linewidth = linewidth
