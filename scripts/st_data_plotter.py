#! /usr/bin/env python
# -*- coding: utf-8 -*-
""" 
Script that creates a scatter plot from one or more ST datasets in matrix format
(genes as columns and spots as rows)

It allows to choose transparency and size for the data points

It allows to normalize the counts using different algorithms

It allows to apply different thresholds

It allows to plot multiple genes (one plot per gene)

@Author Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>
"""

import argparse
import re
from matplotlib import pyplot as plt
from stanalysis.preprocessing import *
import pandas as pd
import numpy as np
import os
import sys
from matplotlib.pyplot import plotting
from scipy.special import loggamma

def normalize(counts, normalization):
    return normalize_data(counts,
                          normalization,
                          center=False,
                          adjusted_log=False,
                          scran_clusters=False)

def filter_data_genes(counts, filter_genes):
    # Extract the list of the genes that must be shown
    genes_to_keep = list()
    if filter_genes:
        for gene in counts.columns:
            for regex in filter_genes:
                if re.fullmatch(regex, gene):
                    genes_to_keep.append(gene)
                    break                         
    else: 
        genes_to_keep = counts.columns
    # Check that we hit some genes
    if len(genes_to_keep) == 0:
        raise RuntimeError("No genes found in the datasets from the " \
                         "list given\n{}\n".format(' '.join([x for x in filter_genes])))
    counts = counts.loc[:,genes_to_keep]
    return counts

# Code snippet taken from Alma Andersson 
# https://github.com/almaan/STDGE/blob/master/enrich.py
def log_binom(n,k):
    """
    Numerically stable binomial coefficient
    """
    n1 = loggamma(n+1)
    d1 = loggamma(k+1)
    d2 = loggamma(n-k + 1)
    return n1 - d1 -d2

# Code snippet taken from Alma Andersson 
# https://github.com/almaan/STDGE/blob/master/enrich.py
def fex(target_set, query_set, full_set, alpha=0.05):
    """
    Fischer Exact test for 3 sets of genes (target, query and full)
    """
    ts = set(target_set)
    qs = set(query_set)
    fs = set(full_set)
    
    qs_and_ts = qs.intersection(ts)
    qs_not_ts = qs.difference(ts)
    ts_not_qs = fs.difference(qs).intersection(ts)
    not_ts_not_qs = fs.difference(qs).difference(ts)
    
    x = np.zeros((2,2))
    x[0,0] = len(qs_and_ts)
    x[0,1] = len(qs_not_ts)
    x[1,0] = len(ts_not_qs)
    x[1,1] = len(not_ts_not_qs)
    
    p1 = log_binom(x[0,:].sum(), x[0,0])
    p2 = log_binom(x[1,:].sum(),x[1,0])
    p3 = log_binom(x.sum(), x[:,0].sum())

    return np.exp(p1 + p2 - p3)

# Code snippet taken from Alma Andersson 
# https://github.com/almaan/STDGE/blob/master/enrich.py
def select_set(counts, names, mass_proportion):
    """
    Select the top G genes which constitutes
    the fraction (mass_proportion) of the counts
    using a cumulative sum distribution
    """    
    sidx = np.fliplr(np.argsort(counts, axis=1)).astype(int)
    cumsum = np.cumsum(np.take_along_axis(counts, sidx, axis=1), axis=1)
    lim = np.max(cumsum, axis=1) * mass_proportion
    lim = lim.reshape(-1,1)
    q = np.argmin(cumsum <= lim, axis=1)
    return [names[sidx[x,0:q[x]]].tolist() for x in range(counts.shape[0])]

# Code snippet taken from Alma Andersson 
# https://github.com/almaan/STDGE/blob/master/enrich.py
def enrichment_score(counts, target_set, mass_proportion=0.90):
    """
    Computes the enrichment score for all
    spots (rows) based on a gene set (target)
    using p-values
    """
    query_all = counts.columns.values
    query_top_list = select_set(counts.values,
                                query_all,
                                mass_proportion = mass_proportion)
    full_set =  query_all.tolist() + target_set
    pvals = [fex(target_set, q, full_set) for q in query_top_list]
    return -np.log(np.array(pvals))

def filter_data(counts, num_exp_genes, num_exp_spots, min_gene_expression):
    if num_exp_spots <= 0.0 and num_exp_genes <= 0.0:
        return counts
    return remove_noise(counts, num_exp_genes, num_exp_spots,
                        min_expression=min_gene_expression)

def compute_plotting_data(counts, names, cutoff_lower, 
                          cutoff_upper, use_global_scale):
    plotting_data = list()
    # counts should be a vector and cutoff should be a percentage (0.0 - 1.0)
    min_gene_exp = counts.quantile(cutoff_lower)
    max_gene_exp = counts.quantile(cutoff_upper)
    print("Using lower cutoff of {} percentile {} of total distribution".format(min_gene_exp, cutoff_lower))
    print("Using upper cutoff of {} percentile {} of total distribution".format(max_gene_exp, cutoff_upper))
    counts[counts < min_gene_exp] = 0
    counts[counts > max_gene_exp] = 0
    vmin_global = counts.min()
    vmax_global = counts.max()
    for i, name in enumerate(names):
        r = re.compile("^{}_".format(i))
        spots = list(filter(r.match, counts.index))
        if len(spots) > 0:
            # Compute the expressions for each spot
            # as the sum of the counts above threshold
            slice = counts.reindex(spots)
            x,y = zip(*map(lambda s: (float(s.split("x")[0].split("_")[1]),
                                      float(s.split("x")[1])), spots))
            # Get the the gene values for each spot
            rel_sum = slice.values
            if not rel_sum.any():
                sys.stdout.write("Warning, the gene given is not expressed in {}\n".format(name))
            vmin = vmin_global if use_global_scale else rel_sum.min() 
            vmax = vmax_global if use_global_scale else rel_sum.max()
            plotting_data.append((x,y,rel_sum,vmin,vmax,name))
    return plotting_data
    
def plot_data(plotting_data, n_col, n_row, dot_size, color_scale,
              xlim, ylim, invert=False, colorbar=False):
    n_col = min(n_col, len(plotting_data))
    n_row = max(int(len(plotting_data) / n_col), 1)
    fig, ax = plt.subplots(n_row, n_col, figsize=(4*n_col, 4*n_row,)) 
    fig.subplots_adjust(left = 0.1, 
                        right = 0.9,
                        bottom = 0.1,
                        top = 0.9,
                        hspace = 0.2, 
                        wspace = 0.4)  
    sc = list()
    for i,a in enumerate(ax.flatten() if n_row > 1 or n_col > 1 else [ax]):
        # Make the actual plot
        data = plotting_data[i]
        s = a.scatter(data[0], data[1], s=dot_size,
                      cmap=plt.get_cmap(color_scale),
                      c=data[2], edgecolor="none",
                      vmin=data[3], vmax=data[4])
        a.set_title(data[5])
        a.set_xlim(xlim)
        a.set_ylim(ylim)
        a.set_aspect('equal')
        if invert:
            a.invert_yaxis()
        a.set_xticks([])
        a.set_yticks([])
        if colorbar:
            fig.colorbar(s, ax=a, fraction=0.046, pad=0.04)
        sc.append(s)
    return fig, ax, sc
    
def update_plot_data(sc, fig, plotting_data, color_scale):
    for i,s in enumerate(sc):
        xy = np.vstack((plotting_data[i][0], plotting_data[i][1]))
        s.set_offsets(xy.T)
        s.set_array(plotting_data[i][2])
        s.set_cmap(plt.get_cmap(color_scale))
    fig.canvas.draw_idle()    
    
def main(counts_table_files,
         cutoff,
         cutoff_upper,
         data_alpha,
         dot_size,
         normalization,
         color_scale,
         filter_genes,
         outdir,
         use_log_scale,
         standard_transformation,
         num_exp_genes,
         num_exp_spots,
         min_gene_expression,
         use_global_scale,
         num_columns,
         xlim,
         ylim,
         invert_y_axes,
         color_bar,
         combine_genes):
         
    #TODO add sanity checks for the thresholds..
    
    if len(counts_table_files) == 0 or \
    any([not os.path.isfile(f) for f in counts_table_files]):
        sys.stderr.write("Error, input file/s not present or invalid format\n")
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
    
    names = [os.path.splitext(os.path.basename(x))[0] for x in counts_table_files]
     
    n_col = min(num_columns, len(counts_table_files))
    n_row = max(int(len(counts_table_files) / n_col), 1)
    
    # Remove noisy spots and genes (Spots are rows and genes are columns)
    counts_filtered = filter_data(counts, num_exp_genes, 
                                  num_exp_spots, min_gene_expression)
    
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
        
    # Filter
    try:
        counts_final = filter_data_genes(counts_normalized, filter_genes)
    except RuntimeError as e:
        sys.stderr.write(str(e) + "\n")
        sys.exit(1)
    
    # Add a column with the combined genes plot
    if combine_genes in "NaiveMean":
        genes_in_set = counts_final.columns.tolist()
        present_genes = (counts_final > 0).sum(axis=1) / len(genes_in_set)
        means = counts_normalized.mean(axis=1)
        counts_final = counts_final.assign(Combined=((counts_final.mean(axis=1) / means) * present_genes).values)
    elif combine_genes in "NaiveSum":
        genes_in_set = counts_final.columns.tolist()
        present_genes = (counts_final > 0).sum(axis=1) / len(genes_in_set)
        sums = counts_normalized.sum(axis=1)
        counts_final = counts_final.assign(Combined=((counts_final.sum(axis=1) / sums) * present_genes).values)        
    elif combine_genes in "CumSum":
        # For the CumSum I need to use all the genes so in order to compute p-values
        genes_in_set = counts_final.columns.tolist()
        counts_final = counts_final.assign(Combined=enrichment_score(counts_normalized, genes_in_set))
        
    # Compute plotting data and plot
    for gene in counts_final.columns:
        print("Plotting gene {}".format(gene))
        plotting_data = compute_plotting_data(counts_final.loc[:,gene], 
                                              names, 
                                              cutoff,
                                              cutoff_upper,
                                              use_global_scale)
                
        # Create a scatter plot for each dataset
        fig, ax, sc = plot_data(plotting_data, n_col, n_row, dot_size, color_scale,
                                xlim, ylim, invert_y_axes, color_bar)
    
        # Save the plot
        fig.suptitle(gene, fontsize=16)
        fig.savefig(os.path.join(outdir,"{}_joint_plot.pdf".format(gene)),
                    format='pdf', dpi=90)
        plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--counts-table-files", required=True, nargs='+', type=str,
                        help="One or more matrices with gene counts per feature/spot (genes as columns)")
    parser.add_argument("--num-exp-genes", default=0.0, metavar="[FLOAT]", type=float,
                        help="The percentage of number of expressed genes (>= --min-gene-expression) a spot\n" \
                        "must have to be kept from the distribution of all expressed genes (0.0 - 1.0) (default: %(default)s)")
    parser.add_argument("--num-exp-spots", default=0.0, metavar="[FLOAT]", type=float,
                        help="The percentage of number of expressed spots a gene\n" \
                        "must have to be kept from the total number of spots (0.0 - 1.0) (default: %(default)s)")
    parser.add_argument("--min-gene-expression", default=1, type=int, metavar="[INT]", choices=range(1, 50),
                        help="The minimum count (number of reads) a gene must have in a spot to be\n"
                        "considered expressed (default: %(default)s)")
    parser.add_argument("--cutoff", default=0.0, metavar="[FLOAT]", type=float,
                        help="The percentage of reads a gene must have in a spot to be counted from" \
                        "the distribution of reads of the gene across all the spots (0.0 - 1.0) (default: %(default)s)")
    parser.add_argument("--cutoff-upper", default=1.0, metavar="[FLOAT]", type=float,
                        help="The percentage of reads a gene should not have in a spot to be counted from" \
                        "the distribution of reads of the gene across all the spots (0.0 - 1.0) (default: %(default)s)")
    parser.add_argument("--data-alpha", type=float, default=1.0, metavar="[FLOAT]",
                        help="The transparency level for the data points, 0 min and 1 max (default: %(default)s)")
    parser.add_argument("--dot-size", type=int, default=20, metavar="[INT]", choices=range(1, 100),
                        help="The size of the dots (default: %(default)s)")
    parser.add_argument("--color-scale", default="YlOrRd", metavar="[STR]", 
                        type=str, 
                        choices=["hot", "binary", "hsv", "Greys", "inferno", "YlOrRd", "bwr", "Spectral", "coolwarm"],
                        help="Different color scales (default: %(default)s)")
    parser.add_argument("--normalization", default="RAW", metavar="[STR]", 
                        type=str, 
                        choices=["RAW", "DESeq2", "REL", "Scran", "CPM"],
                        help="Normalize the counts using:\n" \
                        "RAW = absolute counts\n" \
                        "DESeq2 = DESeq2::estimateSizeFactors(counts)\n" \
                        "Scran = Deconvolution Sum Factors (Marioni et al)\n" \
                        "REL = Each gene count divided by the total count of its spot\n" \
                        "CPM = Each gene count divided by the total count of its spot multiplied by its mean\n" \
                        "(default: %(default)s)")
    parser.add_argument("--standard-transformation", action="store_true", default=False,
                        help="Apply the z-score transformation to each feature (gene)")
    parser.add_argument("--show-genes", help="Regular expression for gene symbols to be shown\n" \
                        "The genes matching the reg-exp will be shown in separate files.\n" \
                        "Can be given several times.",
                        required=True,
                        type=str,
                        nargs='+')
    parser.add_argument("--outdir", default=None, help="Path to output dir")
    parser.add_argument("--use-log-scale", action="store_true", default=False, 
                        help="Plot expression in log space (log2)")
    parser.add_argument("--use-global-scale", action="store_true", default=False, 
                        help="Use a global scale instead of a relative scale")
    parser.add_argument("--num-columns", default=1, type=int, metavar="[INT]",
                        help="The number of columns (default: %(default)s)")
    parser.add_argument("--xlim", default=[1,33], nargs='+', metavar="[FLOAT]", type=float,
                        help="The x axis limits to have equally sized sub-images (default: %(default)s)")
    parser.add_argument("--ylim", default=[1,35], nargs='+', metavar="[FLOAT]", type=float,
                        help="The y axis limits to have equally sized sub-images (default: %(default)s)")
    parser.add_argument("--invert-y-axes", action="store_true", default=True,
                        help="Whether to invert the y axes or not (default True)")
    parser.add_argument("--color-bar", action="store_true", default=True,
                        help="Whether to show the color bar or not (default True)")
    parser.add_argument("--combine-genes", default="None", metavar="[STR]", 
                        type=str, 
                        choices=["None", "NaiveMean", "NaiveSum", "CumSum"],
                        help="Whether to generate a combined plots with the all the genes:\n" \
                        "None = do not create combined plot\n" \
                        "NaiveMean = create combine plot using the mean value of the genes in the spot adjusted by size\n" \
                        "NaiveMean = create combine plot using the sum value of the genes in the spot adjusted by size\n" \
                        "CumSum = create combined plot using a cumulative sum of the genes (0.90) and the Fisher test\n" \
                        "(default: %(default)s)")
    args = parser.parse_args()

    main(args.counts_table_files,
         args.cutoff,
         args.cutoff_upper,
         args.data_alpha,
         args.dot_size,
         args.normalization,
         args.color_scale,
         args.show_genes,
         args.outdir,
         args.use_log_scale,
         args.standard_transformation,
         args.num_exp_genes,
         args.num_exp_spots,
         args.min_gene_expression,
         args.use_global_scale,
         args.num_columns,
         args.xlim,
         args.ylim,
         args.invert_y_axes,
         args.color_bar,
         args.combine_genes)
