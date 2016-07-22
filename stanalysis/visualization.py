""" 
Visualization functions for the st analysis package
"""
import os
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.colors import ListedColormap

color_map = ["red", "green", "blue", "orange", "cyan", "yellow", "orchid", 
             "saddlebrown", "darkcyan", "gray", "darkred", "darkgreen", "darkblue", 
             "antiquewhite", "bisque", "black"]

def scatter_plot(x_points, y_points, colors, 
                 output, alignment=None, cmap=None, title='Scatter', xlabel='X', 
                 ylabel='Y', image=None, alpha=1.0, size=50):
    """ 
    This function makes a scatter plot of a set of points (x,y).
    The alignment matrix is optional to transform the coordinates
    of the points to pixel space.
    If an image is given the image will be set as background.
    The plot will always use a predefine set of colors.
    The plot will be written to a file.
    :param x_points: a list of x coordinates
    :param y_points: a list of y coordinates
    :param colors: a color label for each point
    :param image: the path to the image file
    :param output: the name/path of the output file
    :param alignment: an alignment 3x3 matrix (pass identity to not align)
    :param size: the size of the dots
    :raises: RuntimeError
    """
    assert(len(x_points) == len(y_points) == len(colors))
    # Plot spots with the color class in the tissue image
    fig = plt.figure(figsize=(16,16))
    a = fig.add_subplot(111, aspect='equal')
    base_trans = a.transData
    extent_size = (1,33,35,1)
    # If alignment is None we re-size the image to chip size (1,1,33,35)
    if alignment is not None:
        base_trans = transforms.Affine2D(matrix = alignment) + base_trans
        extent_size = None
    if cmap is None:
        color_list = set(colors)
        color_values = color_map[min(color_list)-1:max(color_list)]
        cmap = ListedColormap(color_values)
    a.scatter(x_points, 
              y_points, 
              c=colors, 
              cmap=cmap, 
              edgecolor="none", 
              s=size, 
              transform=base_trans,
              alpha=alpha)
    if image is not None and os.path.isfile(image):
        img = plt.imread(image)
        a.imshow(img, extent=(1,33,35,1))
    a.set_xlabel(xlabel)
    a.set_ylabel(ylabel)
    a.legend([plt.Line2D((0,1),(0,0), color=x) for x in color_values], 
             color_list, loc="upper right", markerscale=1.0, 
             ncol=1, scatterpoints=1, fontsize=10)
    a.set_title(title, size=20)
    fig.set_size_inches(16, 16)
    fig.savefig(output, dpi=300)