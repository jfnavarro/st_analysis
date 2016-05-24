#! /usr/bin/env python
#@Author Jose Fernandez
""" Visualization functions for the st analysis package"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.colors import ListedColormap

color_map = ["red", "green", "blue", "orange", "cyan", "yellow", "orchid", 
             "saddlebrown", "darkcyan", "gray", "darkred", "darkgreen", "darkblue", 
             "antiquewhite", "bisque", "black"]

def plotSpotsWithImage(x_points, y_points, colors, image, output, alignment, size=50):
    """ 
    This function makes a scatter plot of a set of points (x,y) on top
    of a image. The alignment matrix is optional to transform the coordinates
    of the points to pixel space.
    The plot will always used a predefine set of colors
    The plot will be written to a file
    @param x_points a list of x coordinates
    @param y_points a list of y coordinates
    @param colors a color label for each point
    @param image the path to the image file
    @param output the name/path of the output file
    @param alignment an alignment 3x3 matrix (pass identity to not align)
    @param size the size of the dot
    """
    assert(len(x_points) == len(y_points) == len(colors))
    if not os.path.isfile(image):
        raise RuntimeError("The image given is not valid")
    # Plot spots with the color class in the tissue image
    img = plt.imread(image)
    fig = plt.figure(figsize=(8,8))
    a = fig.add_subplot(111, aspect='equal')
    base_trans = a.transData
    tr = transforms.Affine2D(matrix = alignment) + base_trans
    color_list = set(colors)
    cmap = color_map[min(color_list)-1:max(color_list)]
    a.scatter(x_points, y_points, c=colors, 
              cmap=ListedColormap(cmap), 
              edgecolor="none", s=size, transform=tr)   
    a.imshow(img)
    fig.set_size_inches(16, 16)
    fig.savefig(output, dpi=300)