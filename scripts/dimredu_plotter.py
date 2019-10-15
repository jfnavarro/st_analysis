#! /usr/bin/env python
# -*- coding: utf-8 -*-
""" 
Script that creates takes as input a dimensionality reduced set of coordinates (For each spot),
a matrix counts (spots x genes), a set of of covariates (per spot) and a list of genes
and generates a set of scatter plots:

- dimensionality reduced spots colored by proximity and cluster id
- one plot for each covariate in the dimensionality reduced space
- one plot for each gene in the dimensionality reduced space

The script allows to normalize and filter the data too. 

@Author Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>
"""

import argparse
from matplotlib import pyplot as plt
from stanalysis.preprocessing import *
from stanalysis.analysis import *
import pandas as pd
import numpy as np
import os
import sys
from matplotlib.pyplot import plotting
from stanalysis.visualization import color_map
from matplotlib import cm
from matplotlib import colors 

def plot_tsne(x, y, c, filename, title, xlab, ylab, alpha, size, color_scale, legend, color_bar):
    a = plt.subplot(1, 1, 1)
    # Create the scatter plot
    cmap = cm.get_cmap(color_scale)
    min_v = min(c)
    max_v = max(c)
    sc = a.scatter(x, y, c=c, 
                   edgecolor="none", 
                   cmap=cmap, 
                   s=size,
                   alpha=alpha,
                   vmin=min_v,
                   vmax=max_v)
    # Add legend
    if legend is not None:
        norm = colors.Normalize(vmin=min_v, vmax=max_v)
        unique_c = np.unique(sorted(c))
        a.legend([plt.Line2D((0,1),(0,0), color=cmap(norm(x))) for x in unique_c], 
                 legend, loc="upper right", markerscale=1.0, 
                 ncol=1, scatterpoints=1, fontsize=5)
    # Add x/y labels
    if xlab is not None:
        a.set_xlabel(xlab)
    else:
        a.set_xticklabels([])
        a.axes.get_xaxis().set_visible(False)
    if ylab is not None:
        a.set_ylabel(ylab)
    else:
        a.set_xticklabels([])
        a.axes.get_yaxis().set_visible(False)
    # Add title
    a.set_title(title, size=12)
    # Add color bar
    if color_bar:
        plt.colorbar(sc)
    # Save the plot in a file if the file name is given
    if filename is not None:
        fig = plt.gcf()
        fig.savefig(filename, format='pdf', dpi=180)
        plt.cla()
        plt.close(fig)
    else:
        plt.show()
    
def main(counts_files,
         dim_redu_file,
         meta_file,
         data_alpha,
         dot_size,
         normalization,
         color_scale,
         outdir,
         use_log_scale,
         standard_transformation,
         num_exp_genes,
         num_exp_spots,
         min_gene_expression,
         show_genes):
    
    if len(counts_files) == 0 or \
    any([not os.path.isfile(f) for f in counts_files]):
        sys.stderr.write("Error, input counts file/s not present or invalid format\n")
        sys.exit(1)
    
    if not os.path.isfile(meta_file):
        sys.stderr.write("Error, meta file not present or invalid format\n")
        sys.exit(1)
        
    if not os.path.isfile(dim_redu_file):
        sys.stderr.write("Error, dimensionality reduction file not present or invalid format\n")
        sys.exit(1)
        
    if outdir is None or not os.path.isdir(outdir): 
        outdir = os.getcwd()
    outdir = os.path.abspath(outdir)

    print("Output directory {}".format(outdir))
    print("Input datasets {}".format(" ".join(counts_files))) 
         
    # Merge input sections (Spots are rows and genes are columns)
    counts = aggregate_datatasets(counts_files)
    print("Total number of spots {}".format(len(counts.index)))
    print("Total number of genes {}".format(len(counts.columns)))
    
    # The names of the sections
    names = [os.path.splitext(os.path.basename(x))[0] for x in counts_files]
     
    # Load dimensionality reduction results
    dim_redu = pd.read_csv(dim_redu_file, sep="\t", header=None,
                           index_col=0, engine='c', low_memory=True)

    # Load the meta-info 
    meta = pd.read_csv(meta_file, sep="\t", header=0,
                       index_col=0, engine='c', low_memory=True)
    
    # Remove noisy spots and genes (Spots are rows and genes are columns)
    counts_filtered = filter_data(counts, 
                                  num_exp_genes, 
                                  num_exp_spots, 
                                  min_gene_expression)
    
    # Normalization
    counts_normalized = normalize(counts_filtered, normalization)
    
    # Log the counts
    if use_log_scale:
        print("Transforming datasets to log space...")
        counts_normalized = np.log1p(counts_normalized)
        
    # Apply the z-transformation
    if standard_transformation:
        print("Applying standard transformation...")
        counts_normalized = ztransformation(counts_normalized)
        
    # Make sure the order of rows is the same
    common = np.intersect1d(np.intersect1d(counts_normalized.index, meta.index), dim_redu.index)
    counts_normalized = counts_normalized.loc[common,:]
    meta = meta.loc[common,:]
    dim_redu = dim_redu.loc[common,:]
    
    # Plot dimensionality reduced coordinates for each meta-var
    x = dim_redu.iloc[:,0].to_numpy()
    y = dim_redu.iloc[:,1].to_numpy()
    z = dim_redu.iloc[:,2].to_numpy() if dim_redu.shape[1] == 4 else None
    color_clusters = dim_redu.iloc[:,-1].to_numpy()
    color_rgb = coord_to_rgb(x, y, z)

    # Plot the cluster colors 
    plot_tsne(x, y, c=color_clusters, filename="dim_red_clusters.pdf",
              title="Clusters", xlab=None, ylab=None, alpha=data_alpha, 
              size=dot_size, color_scale="tab20", 
              legend=np.unique(sorted(color_clusters)), color_bar=False)
    # Plot the RGB colors 
    plot_tsne(x, y, c=color_rgb, filename="rgb_colors.pdf",
              title="RGB colors", xlab=None, ylab=None, 
              alpha=data_alpha, size=dot_size, 
              color_scale="tab20", legend=None, color_bar=False)
    # Plot the different variables in metadata
    for var in meta.columns:
        values = meta.loc[:,var].to_numpy()
        unique_vals = np.unique(sorted(values))
        # Convert values to integers
        tmp_dict = dict()
        for i,u in enumerate(unique_vals):
            tmp_dict[u] = i + 1
        vals = [tmp_dict[x] for x in values]
        # Plot the variable
        plot_tsne(x, y, c=vals, filename="{}.pdf".format(var),
                  title=var, xlab=None, ylab=None, 
                  alpha=data_alpha, size=dot_size, color_scale="tab20",
                  legend=unique_vals, color_bar=False)
    # Plot the genes
    if show_genes is not None:
        for gene in show_genes:
            try:
                row_values = counts_normalized.loc[:,gene].to_numpy()
                plot_tsne(x, y, c=row_values, filename="{}.pdf".format(gene),
                          title=gene, xlab=None, ylab=None, 
                          alpha=data_alpha, size=dot_size, color_scale=color_scale,
                          legend=None, color_bar=True)
            except Exception:
                pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--counts-files", required=True, nargs='+', type=str,
                        help="One or more matrices with gene counts per feature/spot (genes as columns)")
    parser.add_argument("--dim-redu-file", required=True, metavar="[STR]", type=str,
                        help="One file containing the dimensionality reduction results\n" \
                        "for each spot (same order as provided in --counts-files)")
    parser.add_argument("--meta-file", required=True, metavar="[STR]", type=str,
                        help="One meta info file (matrix) where rows are the same as the counts matrices\n" \
                        "(same order) and columns are info variables")
    parser.add_argument("--data-alpha", type=float, default=0.8, metavar="[FLOAT]",
                        help="The transparency level for the data points, 0 min and 1 max (default: %(default)s)")
    parser.add_argument("--dot-size", type=int, default=4, metavar="[INT]",
                        help="The size of the dots in the scatter plots (default: %(default)s)")
    parser.add_argument("--color-scale", default="YlOrRd", 
                        type=str, 
                        choices=["viridis", "hot", "binary", "hsv", "Greys", "inferno", "YlOrRd", "bwr", "Spectral", "coolwarm"],
                        help="Different color scales for individual gene plots (default: %(default)s)")
    parser.add_argument("--normalization", default="RAW", 
                        type=str, 
                        choices=["RAW", "REL", "CPM"],
                        help="Normalize the counts using:\n" \
                        "RAW = absolute counts\n" \
                        "REL = Each gene count divided by the total count of its spot\n" \
                        "CPM = Each gene count divided by the total count of its spot multiplied by its mean\n" \
                        "(default: %(default)s)")
    parser.add_argument("--standard-transformation", action="store_true", default=False,
                        help="Apply the z-score transformation to each feature (gene)")
    parser.add_argument("--outdir", default=None, help="Path to output dir")
    parser.add_argument("--use-log-scale", action="store_true", default=False, 
                        help="Plot expression in log space (log2 + 1)")
    parser.add_argument("--num-exp-genes", default=0.01, metavar="[FLOAT]", type=float,
                        help="The percentage of number of expressed genes (>= --min-gene-expression) a spot\n" \
                        "must have to be kept from the distribution of all expressed genes (0.0 - 1.0) (default: %(default)s)")
    parser.add_argument("--num-exp-spots", default=0.01, metavar="[FLOAT]", type=float,
                        help="The percentage of number of expressed spots a gene\n" \
                        "must have to be kept from the total number of spots (0.0 - 1.0) (default: %(default)s)")
    parser.add_argument("--min-gene-expression", default=1, type=float, metavar="[FLOAT]",
                        help="The minimum count (number of reads) a gene must have in a spot to be\n"
                        "considered expressed (default: %(default)s)")
    parser.add_argument("--show-genes", help="List of genes to plot on top of the dimensionality reduction.\n" \
                        "One plot per gene will be created. Can be given several times.",
                        required=False,
                        default=None,
                        type=str,
                        nargs='+')
    args = parser.parse_args()

    main(args.counts_files,
         args.dim_redu_file,
         args.meta_file,
         args.data_alpha,
         args.dot_size,
         args.normalization,
         args.color_scale,
         args.outdir,
         args.use_log_scale,
         args.standard_transformation,
         args.num_exp_genes,
         args.num_exp_spots,
         args.min_gene_expression,
         args.show_genes)

