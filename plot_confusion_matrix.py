#!/usr/bin/env python

__author__ = "Sam Way"
__copyright__ = "Copyright 2011, The QIIME Project"
__credits__ = ["Sam Way", "Dan Malmer", "Will Van Treuren", "Rob Knight"]
__license__ = "GPL"
__version__ = "1.8.0-dev"
__maintainer__ = "Sam Way"
__email__ = "samfway@gmail.com"
__status__ = "Development"

""" Functions to plot a confusion matrix (heat map) 
"""

from numpy import arange
import matplotlib.pyplot as plt

def plot_confusion_matrix(confusion_matrix, labels, output_filename):
    """ Create a confusion matrix plot. 
    	Resulting image is saved to output_filename
    """ 
    max_value = confusion_matrix.max()
    if max_value < 1.0:
        max_value = 1.0 # matrix is normalized 

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(confusion_matrix, cmap=plt.cm.Blues)
    ax.set_xticks(arange(confusion_matrix.shape[0])+0.5, minor=False)
    ax.set_yticks(arange(confusion_matrix.shape[1])+0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xticklabels(labels, minor=False) # add rotation=int to rotate labels
    ax.set_yticklabels(labels, minor=False)
    ax.set_aspect('equal', adjustable='box')  
    ax.set_xlabel('Predicted Labels', fontsize=18)
    ax.set_ylabel('True Labels', fontsize=18)
    ax.xaxis.set_label_position('top')
    heatmap.set_clim(vmin=0,vmax=max_value)
    fig.colorbar(heatmap)
    plt.savefig(output_filename) 
    
