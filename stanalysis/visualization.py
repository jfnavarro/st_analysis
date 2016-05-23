'''
Created on May 23, 2016

@author: josefernandeznavarro
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms

def plotSpotsWithImage(x_points, y_points, colors, image, output, alignment=None):
    """ 
    This function makes a scatter plot of a set of points (x,y) on top
    of a image. The alignment matrix is optional to transform the coordinates
    of the points to pixel space.
    The plot will be written to a file
    @param x_points a list of x coordinates
    @param y_points a list of y coordinates
    @param colors a color label for each point
    @param image the path to the image file
    @param output the name/path of the output file
    @param alignment an optional alignment 3x3 matrix
    """
    assert(len(x_points) == len(y_points) == len(colors))
    if not os.path.isfile(image):
        raise RuntimeError("The image given is not valid")
    # Create alignment matrix 
    alignment_matrix = np.zeros((3,3), dtype=np.float)
    alignment_matrix[0,0] = 1
    alignment_matrix[0,1] = 0
    alignment_matrix[0,2] = 0
    alignment_matrix[1,0] = 0
    alignment_matrix[1,1] = 1
    alignment_matrix[1,2] = 0
    alignment_matrix[2,0] = 0
    alignment_matrix[2,1] = 0
    alignment_matrix[2,2] = 1
    if alignment and len(alignment) == 9:
        alignment_matrix[0,0] = alignment[0]
        alignment_matrix[0,1] = alignment[1]
        alignment_matrix[0,2] = alignment[2]
        alignment_matrix[1,0] = alignment[3]
        alignment_matrix[1,1] = alignment[4]
        alignment_matrix[1,2] = alignment[5]
        alignment_matrix[2,0] = alignment[6]
        alignment_matrix[2,1] = alignment[7]
        alignment_matrix[2,2] = alignment[8]
    # Plot spots with the color class in the tissue image
    img = plt.imread(image)
    fig = plt.figure(figsize=(8,8))
    a = fig.add_subplot(111, aspect='equal')
    base_trans = a.transData
    tr = transforms.Affine2D(matrix = alignment_matrix) + base_trans
    a.scatter(x_points, y_points, c=colors, edgecolor="none", s=50, transform=tr)   
    a.imshow(img)
    fig.set_size_inches(16, 16)
    fig.savefig(output, dpi=300)