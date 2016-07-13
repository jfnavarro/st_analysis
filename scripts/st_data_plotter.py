#! /usr/bin/env python
""" Script that creates a quality scatter plot from a ST-data file in data frame format
(genes as columns and coordinates as rows)
The output will be a .png file with the same name as the input file if no name if given.

It allows to highlight spots with colors using a file with the following format : 

CLASS_NUMBER X Y

It allows to choose transparency for the data points

It allows to pass an image so the spots are plotted on top of it (an alignment file
can be passed along to convert spot coordinates to pixel coordinates)

It allows to normalize the counts using DESeq

It allows to filter out by counts or gene names (following a reg-exp pattern) 
what spots to plot

@Author Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>
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
import os
import sys

def main(input_data,
         image,
         cutoff,
         highlight_barcodes,
         alignment,
         data_alpha,
         highlight_alpha,
         dot_size,
         normalize_counts,
         filter_genes,
         outfile):

    if not os.path.isfile(input_data):
        sys.stderr.write("Error, input file/s not present or invalid format\n")
        sys.exit(1)
    
    if not outfile:
        outfile = "data_plot.png"
        
    # Extract data frame and normalize it if needed (genes as columns)
    norm_counts_table = pd.read_table(input_data, sep="\t", header=0, index_col=0)
    if normalize_counts:
        size_factors = computeSizeFactors(norm_counts_table, function=np.median)
        norm_counts_table = norm_counts_table.div(size_factors) 
                          
    # Extract the list of the genes that must be shown
    genes_to_keep = list()
    if filter_genes:
        for gene in norm_counts_table.columns:
            for regex in filter_genes:
                if re.match(regex, gene):
                    genes_to_keep.append(gene)
                    break                         
    else: 
        genes_to_keep = norm_counts_table.columns
        
    # Compute the expressions for each coordinate
    # as the sum of all spots that pass the thresholds (Gene and counts)
    expression = np.zeros((35, 35), dtype=np.float)
    for spot in norm_counts_table.index:
        tokens = spot.split("x")
        x = tokens[0]
        y = tokens[1]
        sum_count = sum(count for count in norm_counts_table.loc[spot,genes_to_keep] if count > cutoff)
        expression[x, y] = sum_count                
                     
    # Parse the clusters colors if given
    if highlight_barcodes:   
        colors = np.zeros((35,35), dtype=np.int)     
        with open(highlight_barcodes, "r") as filehandler_read:
            for line in filehandler_read.readlines():
                tokens = line.split()
                assert(len(tokens) == 3)
                cluster = int(tokens[0])
                x = float(tokens[1])
                y = float(tokens[2])
                colors[x,y] = cluster

    # Create a scatter plot, if highlight_barcodes is given
    # then plot another scatter plot on the same canvas.
    # If image is given plot it as a background
    fig = plt.figure(figsize=(16,16))
    a = fig.add_subplot(111, aspect='equal')
    alignment_matrix = parseAlignmentMatrix(alignment)
    base_trans = a.transData
    tr = transforms.Affine2D(matrix = alignment_matrix) + base_trans
    # First scatter plot
    x, y = expression.nonzero()
    a.scatter(x, 
              y, 
              c=expression[x, y], 
              cmap=plt.get_cmap("YlOrBr"), 
              edgecolor="none", 
              s=dot_size, 
              transform=tr,
              alpha=data_alpha)
    # Second scatter plot
    if highlight_barcodes:
        x2, y2 = colors.nonzero()
        colors_second = colors[x2, y2]
        color_list = set(colors[x2,y2].tolist())
        cmap = color_map[min(color_list)-1:max(color_list)]
        a.scatter(x2, y2,
                  c=colors_second,
                  cmap=matplotlib.colors.ListedColormap(cmap),
                  edgecolor="none",
                  s=dot_size,
                  transform=tr,
                  alpha=highlight_alpha)
        # plot legend
        a.legend([plt.Line2D((0,1),(0,0), color=x) for x in cmap],
                 color_list, loc="upper right", markerscale=1.0,
                 ncol=1, scatterpoints=1, fontsize=10)
    
    # Plot image as background    
    if image is not None and os.path.isfile(image):
        img = plt.imread(image)
        a.imshow(img)
    # General settings and write to file
    a.set_xlabel("X")
    a.set_ylabel("Y")
    a.set_title("Scatter", size=20)
    fig.set_size_inches(16, 16)
    fig.savefig(outfile, dpi=300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_data", 
                        help="A data frame with counts from ST data (genes as columns)")
    parser.add_argument("--image", default=None, 
                        help="When given the data will plotted on top of the image, \
                        if the alignment matrix is given the data will be aligned")
    parser.add_argument("--cutoff", help="Do not include genes below this reads cut off (default: %(default)s)",
                        type=float, default=0.0, metavar="[FLOAT]", choices=range(0, 100))
    parser.add_argument("--highlight-spots", default=None,
                        help="A file containing spots (x,y) and the class/label they belong to\n CLASS_NUMBER X Y")
    parser.add_argument("--alignment", default=None,
                        help="A file containing the alignment image (array coordinates to pixel coordinates) as a 3x3 matrix")
    parser.add_argument("--data-alpha", type=float, default=1.0, metavar="[FLOAT]", choices=range(0, 1),
                        help="The transparency level for the data points, 0 min and 1 max (default: %(default)s)")
    parser.add_argument("--highlight-alpha", type=float, default=1.0, metavar="[FLOAT]", choices=range(0, 1),
                        help="The transparency level for the highlighted barcodes, 0 min and 1 max (default: %(default)s)")
    parser.add_argument("--dot-size", type=int, default=50, metavar="[INT]", choices=range(10, 100),
                        help="The size of the dots (default: %(default)s)")
    parser.add_argument("--normalize-counts", action="store_true", default=False,
                        help="If given the counts in the imput table will be normalized using DESeq")
    parser.add_argument("--filter-genes", help="Regular expression for \
                        gene symbols to filter out. Can be given several times.",
                        default=None,
                        type=str,
                        action='append')
    parser.add_argument("--outfile", type=str, help="Name of the output file")
    args = parser.parse_args()

    main(args.input_data,
         args.image,
         args.cutoff,
         args.highlight_spots,
         args.alignment,
         float(args.data_alpha),
         float(args.highlight_alpha),
         int(args.dot_size),
         args.normalize_counts,
         args.filter_genes,
         args.outfile)
