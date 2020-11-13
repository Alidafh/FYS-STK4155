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

        self.plot_title = None

        self.x_label = None
        
        self.y_label = None

        self.plots = []


    def create_plot(self, fig_number):
        
        plt.figure(fig_number)
        
        for plot in self.plots:
            plt.plot(plot.x, plot.y, plot.symbol, label=plot.label, linewidth=plot.linewidth)
            
        plt.title(self.plot_title,size=self.label_size)
        plt.xlabel(self.x_label,size=self.label_size, labelpad=self.label_pad)
        plt.ylabel(self.y_label,size=self.label_size, labelpad=self.label_pad)    
            
        plt.legend(fontsize=self.legend_size)
        plt.xticks(fontsize=self.tick_size)
        plt.yticks(fontsize=self.tick_size)    
            
            
    
    def add_plot(self, x, y, symbol, label, linewidth = 3):
        
        self.plots.append(OnePlot(x, y, symbol, label, linewidth = self.line_width))

    
    def reset(self):
        
        self.plots = []
        


class OnePlot:
    
    def __init__(self, x, y, symbol, label, linewidth=3):
        
        self.x = x
        self.y = y
        self.symbol = symbol
        self.label = label
        self.linewidth = linewidth
        