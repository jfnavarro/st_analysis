#! /usr/bin/env python
#@Author Jose Fernandez
""" Script that creates a quality scatter plot from a ST-data file in data frame format.
The output will be a .png file with the same name as the input file.

It allows to highlight spots with colors using a file with the following format : 

CLASS_NUMBER X Y

It allows to choose transparency for the data points

It allows to pass an image so the spots are plotted on top of it (an alignment 
can be passed along to convert spot coordiante to pixel coordinates)

It allows to normalize the counts using DESeq

It allows to filter out by counts or gene names what spots to plot

It allows to highlight genes by regular expression 

If a regular expression for a gene symbol to highlight is provided, the output
image will be stored in a file called *.0.png. It is possible to give several
regular expressions to highlight, by adding another --highlight parameter.
The images from these queries will be stored in files with the number
increasing: *.0.png, *.1.png, *.2.png, etc.
"""

import argparse
import re
import matplotlib
from matplotlib import transforms
from matplotlib import pyplot as plt
from stanalysis.alignment import parseAlignmentMatrix
from stanalysis.visualization import color_map
from stanalysis.normalization import computeSizeFactors
import numpy as np
import pandas as pd
import math
import os
import sys

def main(input_data, 
         highlight_regexes, 
         image,
         cutoff, 
         highlight_barcodes, 
         alignment, 
         data_alpha,
         highlight_alpha,
         dot_size,
         normalize_counts,
         filter_genes,
         highlight_color):

    if not os.path.isfile(input_data):
        sys.stderr.write("Error, input file/s not present or invalid format\n")
        sys.exit(1)
    
    # Some variables
    highlights = []
    expression = np.zeros((33, 33), dtype=np.float)
    colors = np.zeros((33,33), dtype=np.int)
    alignment_matrix = parseAlignmentMatrix(alignment)
    
    # Extract data frame and normalize it if needed
    norm_counts_table = pd.read_table(input_data, sep="\t", header=0)
    if normalize_counts:
        size_factors = computeSizeFactors(norm_counts_table, function=np.median)
        norm_counts_table = norm_counts_table.div(size_factors) 
    
    # Extract values applying thresholds
    genes = norm_counts_table.columns
    for spot in norm_counts_table.index:
        tokens = spot.split("x")
        x = tokens[0]
        y = tokens[1]
        for gene in genes:
            filter_passed = True
            count = norm_counts_table[spot][gene]
            # Discard counts whose gene does not pass filter
            for regex in filter_genes if filter_genes else []:
                if not re.match(regex, gene):
                    filter_passed = False
                    break
            # Add the count if the gene and the count pass the filter
            if filter_passed and count > cutoff:
                expression[x, y] = count
            # Add spots to highlight if they match the reg-exp
            for i, regex in enumerate(highlight_regexes) if highlight_regexes else []:
                if re.search(regex, gene):
                    highlights[i].add((x, y))
                    
     
    # Parse the clusters colors if needed
    if highlight_barcodes:        
        with open(highlight_barcodes, "r") as filehandler_read:
            for line in filehandler_read.readlines():
                tokens = line.split()
                cluster = int(tokens[0])
                x = int(tokens[1])
                y = int(tokens[2])
                colors[x,y] = cluster

    # Create figures (from 1 to up to number of highlight reg expressions)s
    # if hightlight spots are present they will be plotted on top
    x, y = expression.nonzero()
    for i,_ in enumerate(highlight_regexes) if highlight_regexes else [0]:
        f = plt.figure()
        a = f.add_subplot(111, aspect='equal')
        base_trans = a.transData
        tr = transforms.Affine2D(matrix = alignment_matrix) + base_trans
        # First plot the data
        a.scatter(x, y,
                  c=expression[x, y],
                  edgecolor="none",
                  cmap=plt.get_cmap("YlOrBr"),
                  s=dot_size,
                  transform=tr,
                  alpha=data_alpha) 
        
        # Second plot the highlighted spots if applied                       
        if highlight_barcodes:
            x2, y2 = colors.nonzero()
            color_list = set(colors[x2,y2].tolist())
            cmap = color_map[min(color_list)-1:max(color_list)]
            a.scatter(x2, y2,
                      c=colors[x2, y2],
                      cmap=matplotlib.colors.ListedColormap(cmap),
                      edgecolor="none",
                      s=dot_size,
                      transform=tr,
                      alpha=highlight_alpha)
            # TODO add legend with color labels
          
        # Third highlight spots that contain a gene that was present in the reg-exp 
        if highlight_regexes:
            x, y = zip(*highlights[i])
            a.scatter(x, y, 
                      c=highlight_color,
                      edgecolor=highlight_color,
                      s=dot_size + 10,
                      transform=tr,
                      label=highlight_regexes[i])
            
        # Add the image if present  
        if image:
            img = plt.imread(image)
            a.imshow(img)
            
        # Add labels/legend and save the plot to a file
        a.set_xlabel("X")
        a.set_ylabel("Y")
        a.legend()
        a.set_title("Scatter", size=20)
        if highlight_regexes:
            ending = ".{0}.png".format(i)
        else:
            ending = ".png"
        img_file = "data_plot" + ending
        f.set_size_inches(16, 16)
        f.savefig(img_file, dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_data", 
                        help="A data frame with counts from ST data (genes as columns)", required=True)
    parser.add_argument("--highlight", help="Regular expression for \
                        gene symbols to highlight in the quality \
                        scatter plot. Can be given several times.",
                        default=None,
                        type=str,
                        action='append')
    parser.add_argument("--image", default=None, 
                        help="When given the data will plotted on top of the image, \
                        if the alignment matrix is given the data will be aligned")
    parser.add_argument("--cutoff", help="Do not include genes below this reads cut off",
                        type=float, default=0.0)
    parser.add_argument("--highlight-spots", default=None,
                        help="File containing spots (x,y) and the class/label the belong to")
    parser.add_argument("--alignment", default=None,
                        help="A file containing the alignment image (array coordinates to pixel coordinates) as a 3x3 matrix")
    parser.add_argument("--data-alpha", type=float, default=1.0, 
                        help="The transparency level for the data points, 0 min and 1 max")
    parser.add_argument("--highlight-alpha", type=float, default=1.0, 
                        help="The transparency level for the highlighted barcodes, 0 min and 1 max")
    parser.add_argument("--dot-size", type=int, default=50,
                        help="The size of the dots")
    parser.add_argument("--normalize-counts", action="store_true", default=False,
                        help="If given the counts in the imput table will be normalized using DESeq")
    parser.add_argument("--filter-genes", help="Regular expression for \
                        gene symbols to filter out. Can be given several times.",
                        default=None,
                        type=str,
                        action='append')
    parser.add_argument("--highlight-color", default="blue", type=str, help="Color for the highlighted genes")
    args = parser.parse_args()

    main(args.input_data, 
         args.highlight, 
         args.image,
         args.cutoff, 
         args.highlight_spots, 
         args.alignment, 
         float(args.data_alpha),
         float(args.highlight_alpha),
         int(args.dot_size),
         args.normalize_counts,
         args.filter_genes,
         args.highlight_color)
