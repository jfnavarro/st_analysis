#! /usr/bin/env python
""" 
Script that creates a quality scatter plot from a ST-data file in matrix format
(genes as columns and coordinates as rows)

The output will be a .png file with the same name as the input file if no name if given.

It allows to highlight spots with colors using a file with the following format : 

CLASS_NUMBER XxY

It allows to choose transparency for the data points

It allows to pass an image so the spots are plotted on top of it (an alignment file
can be passed along to convert spot coordinates to pixel coordinates)

It allows to normalize the counts different algorithms

It allows to filter out by counts or gene names (following a reg-exp pattern) 
what spots to plot

@Author Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>
"""

import argparse
import re
from matplotlib import pyplot as plt
from stanalysis.visualization import scatter_plot
from stanalysis.preprocessing import *
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
         normalization,
         filter_genes,
         outfile):

    if not os.path.isfile(input_data):
        sys.stderr.write("Error, input file/s not present or invalid format\n")
        sys.exit(1)
    
    if not outfile:
        outfile = "data_plot.png"
        
    # Extract data frame and normalize it if needed (genes as columns)
    norm_counts_table = pd.read_table(input_data, sep="\t", header=0, index_col=0)

    # Normalization
    norm_counts_table = normalize_data(norm_counts_table, normalization)
                         
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
        
    # Compute the expressions for each spot
    # as the sum of all spots that pass the thresholds (Gene and counts)
    x_points = list()
    y_points = list()
    values = list()
    for spot in norm_counts_table.index:
        tokens = spot.split("x")
        assert(len(tokens) == 2)
        x_points.append(float(tokens[0]))
        y_points.append(float(tokens[1]))
        values.append(sum(count for count in 
                          norm_counts_table.loc[spot,genes_to_keep] if count > cutoff))           
                     
    # Parse the clusters colors if given for each spot
    colors = list()
    if highlight_barcodes: 
        with open(highlight_barcodes, "r") as filehandler_read:
            for line in filehandler_read.readlines():
                tokens = line.split()
                assert(len(tokens) == 2 or len(tokens) ==3)
                colors.append(int(tokens[0]))
    
        if len(colors) != len(values):
            sys.stderr.write("Error, the list of spots to highlight does not match the input data\n")
            sys.exit(1)        

    # Create a scatter plot, if highlight_barcodes is given
    # then plot another scatter plot on the same canvas.
    # If image is given plot it as a background
    color = values
    cmap = plt.get_cmap("YlOrBr")
    alpha = data_alpha
    if highlight_barcodes:
        color = colors
        cmap = None
        alpha = highlight_alpha
    scatter_plot(x_points=x_points,
                 y_points=y_points,
                 colors=color,
                 output=outfile,
                 alignment=alignment,
                 cmap=cmap,
                 title='ST Data scatter',
                 xlabel='X',
                 ylabel='Y',
                 image=image,
                 alpha=alpha,
                 size=dot_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_data", 
                        help="A data frame with counts from ST data (genes as columns)")
    parser.add_argument("--image", default=None, 
                        help="When given the data will plotted on top of the image, " \
                        "if the alignment matrix is given the data will be aligned, otherwise it is assumed " \
                        "that the image is cropped to the array boundaries")
    parser.add_argument("--cutoff", help="Do not include genes below this reads cut off per spot (default: %(default)s)",
                        type=float, default=0.0, metavar="[FLOAT]", choices=range(0, 100))
    parser.add_argument("--highlight-spots", default=None,
                        help="A file containing spots (x,y) and the class/label they belong to\n CLASS_NUMBER X Y")
    parser.add_argument("--alignment", default=None,
                        help="A file containing the alignment image (array coordinates to pixel coordinates) " \
                        "as a 3x3 matrix in a tab delimited format. Only useful if the image given is not cropped " \
                        "to the array boundaries of you want to plot the image in original size")
    parser.add_argument("--data-alpha", type=float, default=1.0, metavar="[FLOAT]",
                        help="The transparency level for the data points, 0 min and 1 max (default: %(default)s)")
    parser.add_argument("--highlight-alpha", type=float, default=1.0, metavar="[FLOAT]",
                        help="The transparency level for the highlighted barcodes, 0 min and 1 max (default: %(default)s)")
    parser.add_argument("--dot-size", type=int, default=100, metavar="[INT]", choices=range(10, 500),
                        help="The size of the dots (default: %(default)s)")
    parser.add_argument("--normalization", default="DESeq", metavar="[STR]", 
                        type=str, choices=["RAW", "DESeq", "DESeq2", "DESeq2Log", "EdgeR", "REL"],
                        help="Normalize the counts using RAW(absolute counts) , " \
                        "DESeq, DESeq2, DESeq2Log, EdgeR and " \
                        "REL(relative counts, each gene count divided by the total count of its spot) (default: %(default)s)")
    parser.add_argument("--show-genes", help="Regular expression for gene symbols to be shown",
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
         args.normalization,
         args.filter_genes,
         args.outfile)
