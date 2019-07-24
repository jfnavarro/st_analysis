""" 
Alignment functions for the ST Analysis packages.
Alignment refers to the matrix that transforms
spots in array coordinates to image pixel coordinates.
"""
import numpy as np
import os

def parseAlignmentMatrix(alignment_file):
    """ 
    Takes a file as input that contains 
    the values of a 3x3 affine matrix in one line
    as :
    a11 a12 a13 a21 a22 a23 a31 a32 a33
    and returns a 3x3 matrix with the parsed elements
    :param alignment_file: a file containing the 9 elements of a 3x3 matrix
    :return: a 3x3 matrix (default identify if error happens)
    """
    alignment_matrix = np.identity(3)
    if alignment_file is None or not os.path.isfile(alignment_file):
        return alignment_matrix
    with open(alignment_file, "r") as filehandler:
        line = filehandler.readline()
        tokens = line.split()
        assert(len(tokens) == 9)
        alignment_matrix[0,0] = float(tokens[0])
        alignment_matrix[1,0] = float(tokens[1])
        alignment_matrix[2,0] = float(tokens[2])
        alignment_matrix[0,1] = float(tokens[3])
        alignment_matrix[1,1] = float(tokens[4])
        alignment_matrix[2,1] = float(tokens[5])
        alignment_matrix[0,2] = float(tokens[6])
        alignment_matrix[1,2] = float(tokens[7])
        alignment_matrix[2,2] = float(tokens[8])
    return alignment_matrix