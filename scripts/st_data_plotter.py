#! /usr/bin/env python
# -*- coding: utf-8 -*-
""" 
Script that creates a scatter plot from one or more ST datasets in matrix format
(genes as columns and spots as rows)

It allows to choose transparency and size for the data points

It allows to pass images so the spots are plotted on top of it (an alignment file
can be passed along to convert spot coordinates to pixel coordinates)

It allows to normalize the counts using different algorithms

It allows to apply different thresholds

It allows to filter out by gene counts or gene names (following a reg-exp pattern) 
what spots to plot

@Author Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>
"""

import argparse
import re
from matplotlib import pyplot as plt
from stanalysis.visualization import scatter_plot
from stanalysis.preprocessing import *
from stanalysis.alignment import parseAlignmentMatrix
import pandas as pd
import numpy as np
import os
import sys

def main(counts_table_files,
         image_files,
         alignment_files,
         cutoff,
         data_alpha,
         dot_size,
         normalization,
         filter_genes,
         outdir,
         use_log_scale,
         num_exp_genes,
         num_exp_spots,
         min_gene_expression,
         joint_plot,
         use_global_scale,
         num_columns):

    if len(counts_table_files) == 0 or \
    any([not os.path.isfile(f) for f in counts_table_files]):
        sys.stderr.write("Error, input file/s not present or invalid format\n")
        sys.exit(1)
    
    if image_files is not None and len(image_files) > 0 and \
    len(image_files) != len(counts_table_files):
        sys.stderr.write("Error, the number of images given as " \
        "input is not the same as the number of datasets\n")
        sys.exit(1)           
   
    if alignment_files is not None and len(alignment_files) > 0 \
    and len(alignment_files) != len(image_files):
        sys.stderr.write("Error, the number of alignments given as " \
        "input is not the same as the number of images\n")
        sys.exit(1)
    
    if outdir is None or not os.path.isdir(outdir): 
        outdir = os.getcwd()
    outdir = os.path.abspath(outdir)

    print("Output directory {}".format(outdir))
    print("Input datasets {}".format(" ".join(counts_table_files))) 
         
    # Merge input datasets (Spots are rows and genes are columns)
    counts = aggregate_datatasets(counts_table_files)
    print("Total number of spots {}".format(len(counts.index)))
    print("Total number of genes {}".format(len(counts.columns)))

    # Remove noisy spots and genes (Spots are rows and genes are columns)
    counts = remove_noise(counts, num_exp_genes / 100.0, num_exp_spots / 100.0,
                          min_expression=min_gene_expression)
    
    # Normalization
    print("Computing per spot normalization...")
    counts = normalize_data(counts, 
                            normalization,
                            center=False,
                            adjusted_log=False)
                         
    # Extract the list of the genes that must be shown
    genes_to_keep = list()
    if filter_genes:
        for gene in counts.columns:
            for regex in filter_genes:
                if re.match(regex, gene):
                    genes_to_keep.append(gene)
                    break                         
    else: 
        genes_to_keep = counts.columns
    
    if len(genes_to_keep) == 0:
        sys.stderr.write("Error, no genes found with the reg-exp given\n")
        sys.exit(1)        
    counts = counts.loc[:,genes_to_keep]
    counts = counts.loc[(counts!=0).any(axis=1)]
    
    # Create a scatter plot for each dataset
    print("Plotting data...")
    n_col = min(num_columns, len(counts_table_files)) if joint_plot else 1
    n_row = max(int(len(counts_table_files) / n_col), 1) if joint_plot else 1
    if joint_plot:
        print("Generating a multiplot of {} rows and {} columns".format(n_row, n_col))
    total_spots = counts.index
    global_sum = np.log2(counts[counts > cutoff]).sum(1) if use_log_scale else counts[counts > cutoff].sum(1)
    vmin_global = global_sum.min()
    vmax_global = global_sum.max()
    for i, name in enumerate(counts_table_files):
        spots = list(filter(lambda x:'{}_'.format(i) in x, total_spots))
        # Compute the expressions for each spot
        # as the sum of the counts above threshold
        slice = counts.reindex(spots)
        slice = slice.loc[(slice!=0).any(axis=1)]
        x,y = zip(*map(lambda s: (float(s.split("x")[0].split("_")[1]),float(s.split("x")[1])), spots))
        rel_sum = np.log2(slice[slice > cutoff]).sum(1) if use_log_scale else slice[slice > cutoff].sum(1)
        if not rel_sum.any():
            sys.stdout.write("Warning, the gene/s given are not expressed in {}\n".format(name))
        vmin = vmin_global if use_global_scale else rel_sum.min() 
        vmax = vmax_global if use_global_scale else rel_sum.max()
        
        # Retrieve alignment matrix and image if any
        image = image_files[i] if image_files is not None else None
        alignment = alignment_files[i] if alignment_files is not None else None
                
        # alignment_matrix will be identity if alignment file is None
        alignment_matrix = parseAlignmentMatrix(alignment) 
    
        # Create a scatter plot for the gene data
        # If image is given plot it as a background
        outfile = os.path.join(outdir, "{}.pdf".format(
            os.path.splitext(os.path.basename(name))[0])) if not joint_plot else None
        scatter_plot(x_points=x,
                     y_points=y,
                     colors=rel_sum,
                     output=outfile,
                     alignment=alignment_matrix,
                     cmap=plt.get_cmap("YlOrBr"),
                     title=name,
                     xlabel=None,
                     ylabel=None,
                     image=image,
                     alpha=data_alpha,
                     size=dot_size,
                     show_legend=False,
                     show_color_bar=False,
                     vmin=vmin,
                     vmax=vmax,
                     n_col=n_col,
                     n_row=n_row,
                     n_index=i+1 if joint_plot else 1)
    
    if joint_plot:
        #plt.subplots_adjust(left=0.125, 
        #                    bottom=0.9, 
        #                    right=0.1, 
        #                    top=0.9,
        #                    wspace=0.3, 
        #                    hspace=0.3)
        fig = plt.gcf()
        fig.savefig(os.path.join(outdir,"joint_plot.pdf"), format='pdf', dpi=180)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--counts-table-files", required=True, nargs='+', type=str,
                        help="One or more matrices with gene counts per feature/spot (genes as columns)")
    parser.add_argument("--alignment-files", default=None, nargs='+', type=str,
                        help="One or more tab delimited files containing and alignment matrix for the images as\n" \
                        "\t a11 a12 a13 a21 a22 a23 a31 a32 a33\n" \
                        "Only useful is the image has extra borders, for instance not cropped to the array corners\n" \
                        "or if you want the keep the original image size in the plots.")
    parser.add_argument("--image-files", default=None, nargs='+', type=str,
                        help="When provided the data will plotted on top of the image\n" \
                        "It can be one ore more, ideally one for each input dataset\n " \
                        "It is desirable that the image is cropped to the array\n" \
                        "corners otherwise an alignment file is needed")
    parser.add_argument("--num-exp-genes", default=1, metavar="[FLOAT]", type=float,
                        help="The percentage of number of expressed genes (>= --min-gene-expression) a spot\n" \
                        "must have to be kept from the distribution of all expressed genes (default: %(default)s)")
    parser.add_argument("--num-exp-spots", default=1, metavar="[FLOAT]", type=float,
                        help="The percentage of number of expressed spots a gene\n" \
                        "must have to be kept from the total number of spots (default: %(default)s)")
    parser.add_argument("--min-gene-expression", default=1, type=int, metavar="[INT]", choices=range(1, 50),
                        help="The minimum count (number of reads) a gene must have in a spot to be\n"
                        "considered expressed (default: %(default)s)")
    parser.add_argument("--cutoff", 
                        help="Do not include genes that fall below this reads cut off per spot (default: %(default)s)",
                        type=float, default=0.0, metavar="[FLOAT]", choices=range(0, 100))
    parser.add_argument("--data-alpha", type=float, default=1.0, metavar="[FLOAT]",
                        help="The transparency level for the data points, 0 min and 1 max (default: %(default)s)")
    parser.add_argument("--dot-size", type=int, default=20, metavar="[INT]", choices=range(1, 100),
                        help="The size of the dots (default: %(default)s)")
    parser.add_argument("--normalization", default="RAW", metavar="[STR]", 
                        type=str, 
                        choices=["RAW", "DESeq2", "REL", "Scran"],
                        help="Normalize the counts using:\n" \
                        "RAW = absolute counts\n" \
                        "DESeq2 = DESeq2::estimateSizeFactors(counts)\n" \
                        "Scran = Deconvolution Sum Factors (Marioni et al)\n" \
                        "REL = Each gene count divided by the total count of its spot\n" \
                        "(default: %(default)s)")
    parser.add_argument("--show-genes", help="Regular expression for gene symbols to be shown\n" \
                        "If given only the genes matching the reg-exp will be shown.\n" \
                        "Can be given several times.",
                        default=None,
                        type=str,
                        action='append')
    parser.add_argument("--outdir", default=None, help="Path to output dir")
    parser.add_argument("--use-log-scale", action="store_true", default=False, 
                        help="Plot expression in log space (log2)")
    parser.add_argument("--joint-plot", action="store_true", default=False, 
                        help="Generate one figure for all the datasets instead of one figure per dataset.")
    parser.add_argument("--use-global-scale", action="store_true", default=False, 
                        help="Use a global scale instead of a relative one when plotting several datasets.")
    parser.add_argument("--num-columns", default=1, type=int, metavar="[INT]",
                        help="The number of columns when using --joint-plot (default: %(default)s)")
    args = parser.parse_args()

    main(args.counts_table_files,
         args.image_files,
         args.alignment_files,
         args.cutoff,
         args.data_alpha,
         args.dot_size,
         args.normalization,
         args.show_genes,
         args.outdir,
         args.use_log_scale,
         args.num_exp_genes,
         args.num_exp_spots,
         args.min_gene_expression,
         args.joint_plot,
         args.use_global_scale,
         args.num_columns)
