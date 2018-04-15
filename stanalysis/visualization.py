""" 
Visualization functions for the st analysis package
"""
import os
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import transforms
from matplotlib.colors import ListedColormap
import numpy as np

color_map = ["red", "green", "blue", "orange", "cyan", "yellow", "orchid", 
             "saddlebrown", "darkcyan", "gray", "darkred", "darkgreen", "darkblue", 
             "antiquewhite", "bisque", "black", "slategray", "gold", "floralwhite",
             "aliceblue", "plum", "cadetblue", "coral", "olive", "khaki", "lightsalmon"]

def volcano(dea_results, fdr, outfile):
    """ Generates a volcano plot for the given DEA results
    :param dea_results: a data frame that must contains a padj 
    and a log2FoldChange columns
    :param fdr: the fdr threshold to apply (0-1)
    :param outfile: the name of the output file
    """
    fig, a = plt.subplots(figsize=(30, 30))
    colors = ["red" if p <= fdr else "blue" for p in dea_results["padj"]]
    x_points = dea_results["log2FoldChange"]
    y_points = -np.log10(dea_results["pvalue"] + np.finfo(np.float32).eps)
    x_points_conf = dea_results.ix[dea_results["padj"] <= fdr]["log2FoldChange"]
    y_points_conf = -np.log10(dea_results.ix[dea_results["padj"] <= fdr]["pvalue"] + + np.finfo(np.float32).eps)
    names_conf = dea_results.ix[dea_results["padj"] <= fdr].index
    # Scale axes
    OFFSET = 0.1
    a.set_xlim([min(x_points) - OFFSET, max(x_points) + OFFSET])
    a.set_ylim([min(y_points) - OFFSET, max(y_points) + OFFSET])
    a.set_xlabel("Log2FoldChange")
    a.set_ylabel("-log10(pvalue)")
    a.set_title("Volcano plot", size=10)
    a.scatter(x_points, y_points, c=colors, edgecolor="none")  
    for x,y,text in zip(x_points_conf,y_points_conf,names_conf):
        a.text(x,y,text,size="x-small")
    fig.savefig(outfile, dpi=300)
    
def histogram(x_points, output, title="Histogram", xlabel="X", color="blue"):
    """ This function generates a simple density histogram
    with the points given as input.
    :param x_points: a list of x coordinates
    :param title: the title for the plot
    :param xlabel: the name of the X label
    :param output: the name/path of the output file
    :param color: the color for the histogram
    """
    fig = plt.figure()

    # the histogram of the data
    n, bins, patches = plt.hist(x_points, bins="auto", 
                                normed=False, facecolor=color, alpha=1.0)
    
    mean = np.mean(x_points)
    std_dev = np.std(x_points)
    # add a 'best fit' line
    y = mlab.normpdf(bins, mean, std_dev)
    plt.plot(bins, y, 'r--', linewidth=1)
    # generate plot
    plt.xlabel(xlabel)
    plt.ylabel("Occurrences")
    plt.title(title)

    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    fig.savefig("{}.pdf".format(os.path.splitext(os.path.basename(output))[0]), 
                format='pdf', dpi=300)
    
def scatter_plot3d(x_points, y_points, z_points, output=None,
                   colors=None, cmap=None, title='Scatter', xlabel='X', 
                   ylabel='Y', zlabel="Z", alpha=1.0, size=10, vmin=None, vmax=None):
    """ 
    This function makes a scatter 3d plot of a set of points (x,y,z).
    The plot will always use a predefine set of colors unless specified otherwise.
    The plot will be written to a file.
    :param x_points: a list of x coordinates
    :param y_points: a list of y coordinates
    :param z_points: a list of z coordinates (optional)
    :param output: the name/path of the output file
    :param colors: a color label for each point (can be None)
    :param cmap: Matplotlib color mapping object (optional)
    :param title: the title for the plot
    :param xlabel: the name of the X label
    :param ylabel: the name of the Y label
    :param alpha: the alpha transparency level for the dots
    :param size: the size of the dots
    :raises: RuntimeError
    """
    # Plot spots with the color class in the tissue image
    fig = plt.figure()
    a = plt.subplot(projection="3d")
    color_values = None
    unique_colors = set(colors)
    if cmap is None and colors is not None:
        color_values = [color_map[i] for i in unique_colors]
        colors = [color_map[i] for i in colors]
    elif colors is None:
        colors = "blue"
    a.scatter(x_points, 
              y_points,
              z_points,
              c=colors, 
              cmap=cmap, 
              edgecolor="none", 
              s=size,
              alpha=alpha,
              vmin=vmin,
              vmax=vmax)
    a.set_xlabel(xlabel)
    a.set_ylabel(ylabel)
    a.set_zlabel(zlabel)
    if color_values is not None:
        a.legend([plt.Line2D((0,1),(0,0), color=x) for x in color_values], 
                 unique_colors, loc="upper right", markerscale=1.0, 
                 ncol=1, scatterpoints=1, fontsize=5)
    a.set_title(title, size=10)
    # Save or show the plot
    if output is not None:
        fig.savefig("{}.pdf".format(os.path.splitext(os.path.basename(output))[0]), 
                    format='pdf', dpi=300)
    else:
        fig.show()
   
def grid_plot(x_points, y_points, colors, output=None, alignment=None):
     return
 
def scatter_plot(x_points, y_points, output=None, colors=None,
                 alignment=None, cmap=None, title='Scatter', xlabel='X', 
                 ylabel='Y', image=None, alpha=1.0, size=10, 
                 show_legend=True, show_color_bar=False, vmin=None, vmax=None):
    """ 
    This function makes a scatter plot of a set of points (x,y).
    The alignment matrix is optional to transform the coordinates
    of the points to pixel space.
    If an image is given the image will be set as background.
    The plot will always use a predefine set of colors.
    The plot will be written to a file.
    :param x_points: a list of x coordinates
    :param y_points: a list of y coordinates
    :param output: the name/path of the output file
    :param colors: a color int value label for each point (can be None)
    :param alignment: an alignment 3x3 matrix (pass identity to not align)
    :param cmap: Matplotlib color mapping object (optional)
    :param title: the title for the plot
    :param xlabel: the name of the X label
    :param ylabel: the name of the Y label
    :param image: the path to the image file
    :param alpha: the alpha transparency level for the dots
    :param size: the size of the dots
    :param show_legend: True draws a legend with the unique colors
    :param show_color_bar: True draws the color bar distribution
    :raises: RuntimeError
    """
    # Plot spots with the color class in the tissue image
    fig, a = plt.subplots()
    base_trans = a.transData
    # Extend (left, right, bottom, top)
    # The location, in data-coordinates, of the lower-left and upper-right corners. 
    # If None, the image is positioned such that the pixel centers fall on zero-based (row, column) indices.
    extent_size = [1,33,35,1]
    # If alignment is None we re-size the image to chip size (1,1,33,35)
    # Otherwise we keep the image intact and apply the 3x3 transformation
    if alignment is not None and not np.array_equal(alignment, np.identity(3)):
        base_trans = transforms.Affine2D(matrix = alignment) + base_trans
        extent_size = None
    # We convert the list of color int values to color labels
    color_values = None
    unique_colors = set(colors)
    if cmap is None and colors is not None:
        color_values = [color_map[i] for i in unique_colors]
        colors = [color_map[i] for i in colors]
    elif colors is None:
        colors = "blue"
    # Create the scatter plot      
    sc = a.scatter(x_points, y_points, c=colors, edgecolor="none", 
                   cmap=cmap, s=size, transform=base_trans, alpha=alpha,
                   vmin=vmin, vmax=vmax)
    # Plot the image
    if image is not None and os.path.isfile(image):
        img = plt.imread(image)
        a.imshow(img, extent=extent_size)
    # Add labels and title
    a.set_xlabel(xlabel)
    a.set_ylabel(ylabel)
    a.set_title(title, size=10)
    # Add legend
    if color_values is not None and show_legend:
        a.legend([plt.Line2D((0,1),(0,0), color=x) for x in color_values], 
                 unique_colors, loc="upper right", markerscale=1.0, 
                 ncol=1, scatterpoints=1, fontsize=5)
    # Add color bar
    if colors is not None and show_color_bar:
        plt.colorbar(sc)
    # Save or show the plot
    if output is not None:
        fig.savefig("{}.pdf".format(os.path.splitext(os.path.basename(output))[0]), 
                    format='pdf', dpi=180)
    else:
        fig.show()