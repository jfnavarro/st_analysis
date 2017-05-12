#! /usr/bin/env python
""" 
Script that creates a quality scatter plot from a ST-data file in matrix format
(genes as columns and spots as rows)

The output will be a .png file with the same name as the input file if no name if given.

It allows to highlight spots with colors using a file with the following format : 

INTEGER XxY

When highlighting spots a new file will be created with the highlighted
spots.

It allows to choose transparency for the data points

It allows to pass an image so the spots are plotted on top of it (an alignment file
can be passed along to convert spot coordinates to pixel coordinates)

It allows to normalize the counts usins different algorithms

It allows to filter out by gene counts or gene names (following a reg-exp pattern) 
what spots to plot

@Author Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>
"""

import argparse
import re
from matplotlib import pyplot as plt
from stanalysis.visualization import scatter_plot
from stanalysis.preprocessing import *
import pandas as pd
import numpy as np
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
         outfile,
         use_log_scale,
         title):

    if not os.path.isfile(input_data):
        sys.stderr.write("Error, input file/s not present or invalid format\n")
        sys.exit(1)
    
    if not outfile:
        outfile = "data_plot.pdf"
        
    # Extract data frame and normalize it if needed (genes as columns)
    counts_table = pd.read_table(input_data, sep="\t", header=0, index_col=0)

    # Normalization
    norm_counts_table = normalize_data(counts_table, normalization)
                         
    # Extract the list of the genes that must be shown
    genes_to_keep = list()
    if filter_genes:
        for gene in norm_counts_table.columns:
            for regex in filter_genes:
                if re.match(regex, gene):
                    print gene
                    genes_to_keep.append(gene)
                    break                         
    else: 
        genes_to_keep = norm_counts_table.columns
    
    if len(genes_to_keep) == 0:
        sys.stderr.write("Error, no genes found with the reg-exp given\n")
        sys.exit(1)        
    
    # Compute the expressions for each spot
    # as the sum of all spots that pass the thresholds (Gene and counts)
    x_points = list()
    y_points = list()
    colors = list()
    for spot in norm_counts_table.index:
        tokens = spot.split("x")
        assert(len(tokens) == 2)
        exp = sum(count for count in norm_counts_table.loc[spot,genes_to_keep] if count > cutoff)
        if exp > 0.0:
            x_points.append(float(tokens[0]))
            y_points.append(float(tokens[1]))
            if use_log_scale: exp = np.log2(exp)
            colors.append(exp)           
       
    if len(colors) == 0:
        sys.stderr.write("Error, the gene/s given are not expressed in this dataset\n")
        sys.exit(1)   
                
    # If highlight barcodes is given then
    # parse the spots and their color and plot
    # them on top of the image if given
    if highlight_barcodes:
        colors_highlight = list()
        x_points_highlight = list()
        y_points_highlight = list()
        with open(highlight_barcodes, "r") as filehandler_read:
            for line in filehandler_read.readlines():
                tokens = line.split()
                assert(len(tokens) == 2)
                tokens2 = tokens[1].split("x")
                assert(len(tokens2) == 2)
                x_points_highlight.append(float(tokens2[0]))
                y_points_highlight.append(float(tokens2[1]))
                colors_highlight.append(int(tokens[0]))
        scatter_plot(x_points=x_points_highlight,
                     y_points=y_points_highlight,
                     colors=colors_highlight,
                     output="{}_{}".format("highlight",outfile),
                     alignment=alignment,
                     cmap=None,
                     title=title,
                     xlabel='X',
                     ylabel='Y',
                     image=image,
                     alpha=highlight_alpha,
                     size=dot_size,
                     show_legend=True,
                     show_color_bar=False)     

    # Create a scatter plot for the gene data
    # If image is given plot it as a background
    scatter_plot(x_points=x_points,
                 y_points=y_points,
                 colors=colors,
                 output=outfile,
                 alignment=alignment,
                 cmap=plt.get_cmap("YlOrBr"),
                 title=title,
                 xlabel='X',
                 ylabel='Y',
                 image=image,
                 alpha=data_alpha,
                 size=dot_size,
                 show_legend=False,
                 show_color_bar=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("input_data", 
                        help="A data frame with counts from ST data (genes as columns)")
    parser.add_argument("--image", default=None, 
                        help="When given an image the data will plotted on top of the image\n" \
                        "if the alignment matrix is given the data will be aligned, otherwise it is assumed\n" \
                        "that the image is cropped to the array boundaries")
    parser.add_argument("--cutoff", 
                        help="Do not include genes that falls below this reads cut off per spot (default: %(default)s)",
                        type=float, default=0.0, metavar="[FLOAT]", choices=range(0, 100))
    parser.add_argument("--highlight-spots", default=None,
                        help="A file containing spots (XxY) and the class/label they belong to\n INT XxY")
    parser.add_argument("--alignment", default=None,
                        help="A file containing the alignment image (array coordinates to pixel coordinates)\n" \
                        "as a 3x3 matrix in a tab delimited format. Only useful if the image given is not cropped\n" \
                        "to the array boundaries of you want to plot the image in original size")
    parser.add_argument("--data-alpha", type=float, default=1.0, metavar="[FLOAT]",
                        help="The transparency level for the data points, 0 min and 1 max (default: %(default)s)")
    parser.add_argument("--highlight-alpha", type=float, default=1.0, metavar="[FLOAT]",
                        help="The transparency level for the highlighted barcodes, 0 min and 1 max (default: %(default)s)")
    parser.add_argument("--dot-size", type=int, default=20, metavar="[INT]", choices=range(1, 100),
                        help="The size of the dots (default: %(default)s)")
    parser.add_argument("--normalization", default="RAW", metavar="[STR]", 
                        type=str, 
                        choices=["RAW", "DESeq2", "DESeq2Linear", "DESeq2PseudoCount", 
                                 "DESeq2SizeAdjusted", "REL", "TMM", "RLE", "Scran"],
                        help="Normalize the counts using:\n" \
                        "RAW = absolute counts\n" \
                        "DESeq2 = DESeq2::estimateSizeFactors(counts)\n" \
                        "DESeq2PseudoCount = DESeq2::estimateSizeFactors(counts + 1)\n" \
                        "DESeq2Linear = DESeq2::estimateSizeFactors(counts, linear=TRUE)\n" \
                        "DESeq2SizeAdjusted = DESeq2::estimateSizeFactors(counts + lib_size_factors)\n" \
                        "RLE = EdgeR RLE * lib_size\n" \
                        "TMM = EdgeR TMM * lib_size\n" \
                        "Scran = Deconvolution Sum Factors\n" \
                        "REL = Each gene count divided by the total count of its spot\n" \
                        "(default: %(default)s)")
    parser.add_argument("--show-genes", help="Regular expression for gene symbols to be shown\n" \
                        "If given only the genes matching the reg-exp will be shown.\n" \
                        "Can be given several times.",
                        default=None,
                        type=str,
                        action='append')
    parser.add_argument("--title", help="The title to show in the plot.", default="ST Data scatter", type=str)
    parser.add_argument("--outfile", type=str, help="Name of the output file")
    parser.add_argument("--use-log-scale", action="store_true", default=False, help="Use log2(counts + 1) values")
    args = parser.parse_args()

    main(args.input_data,
         args.image,
         args.cutoff,
         args.highlight_spots,
         args.alignment,
         args.data_alpha,
         args.highlight_alpha,
         args.dot_size,
         args.normalization,
         args.show_genes,
         args.outfile,
         args.use_log_scale,
         args.title)
