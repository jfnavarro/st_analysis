#! /usr/bin/env python
# -*- coding: utf-8 -*-
""" 
Script that creates a scatter plot from one or more ST datasets in matrix format
(genes as columns and spots as rows)

It allows to choose transparency and size for the data points

It allows to normalize the counts using different algorithms

It allows to apply different thresholds

It allows to plot multiple genes (one plot per gene)

It allows to plot combined gene sets (one plot for the whole set)

It allows to plot clusters (one cluster per spot) 

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

def get_spot_coordinates(spots):
    has_index = "_" in spots[0]
    return zip(*map(lambda s: (float(s.split("x")[0].split("_")[1] if has_index else s.split("x")[0]),
                               float(s.split("x")[1])), spots))
    
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
        r = re.compile("^{}_".format(i + 1))
        # Filter spot by index (section) unless only one section is given
        spots = list(filter(r.match, counts.index)) if len(names) > 1 else counts.index
        if len(spots) > 0:
            # Compute the expressions for each spot
            # as the sum of the counts over the gene
            rel_sum = counts.reindex(spots).values
            x,y = get_spot_coordinates(spots)
            if not rel_sum.any():
                sys.stdout.write("Warning, the gene given is not expressed in {}\n".format(name))
            vmin = vmin_global if use_global_scale else rel_sum.min() 
            vmax = vmax_global if use_global_scale else rel_sum.max()
            plotting_data.append((x,y,rel_sum,vmin,vmax,name))
    return plotting_data
    
def compute_plotting_data_clusters(counts, names, clusters):
    plotting_data = list()
    vmin_global = int(clusters.iloc[:,0].min())
    vmax_global = int(clusters.iloc[:,0].max())
    for i, name in enumerate(names):
        r = re.compile("^{}_".format(i + 1))
        # Filter spot by index (section) unless only one section is given
        spots = list(filter(r.match, counts.index)) if len(names) > 1 else counts.index
        if len(spots) > 0:
            x,y = get_spot_coordinates(spots)
            c = np.ravel(clusters.loc[spots,:].values.astype(int))
            plotting_data.append((x,y,c,vmin_global,vmax_global,name))
    return plotting_data

def plot_data(plotting_data, n_col, n_row, dot_size, data_alpha,
              color_scale, xlim, ylim, invert=False, colorbar=False):
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
                      alpha=data_alpha,
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
    
def main(counts_table_files,
         cutoff,
         cutoff_upper,
         data_alpha,
         dot_size,
         normalization,
         color_scale,
         color_scale_clusters,
         filter_genes,
         clusters_file,
         gene_family,
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
         disable_invert_y_axes,
         disable_color_bar,
         combine_genes):
         
    if cutoff_upper <= cutoff:
        sys.stderr.write("Error, incorrect cut-off values {}\n".format(cutoff))
        sys.exit(1)
        
    if dot_size < 0:
        sys.stderr.write("Error, incorrect dot size {}\n".format(dot_size))
        sys.exit(1)
        
    if data_alpha < 0 or data_alpha > 1:
        sys.stderr.write("Error, incorrect alpha value {}\n".format(data_alpha))
        sys.exit(1)
        
    if len(counts_table_files) == 0 or \
    any([not os.path.isfile(f) for f in counts_table_files]):
        sys.stderr.write("Error, input counts not present or invalid format {}.\n".format('\n'.join(counts_table_files)))
        sys.exit(1)
        
    if gene_family and \
    any([not os.path.isfile(f) for f in counts_table_files]):
        sys.stderr.write("Error, input gene family not present or invalid format {}.\n".format('\n'.join(counts_table_files)))
        sys.exit(1)
        
    if num_exp_genes < 0 or num_exp_genes > 1:
        sys.stderr.write("Error, invalid number of expressed genes {}\n".format(num_exp_genes))
        sys.exit(1)
         
    if num_exp_spots < 0 or num_exp_spots > 1:
        sys.stderr.write("Error, invalid number of expressed genes {}\n".format(num_exp_spots))
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
    
    # Get the names of the datasets
    names = [os.path.splitext(os.path.basename(x))[0] for x in counts_table_files]
     
    # Compute number of columns/rows
    n_col = min(num_columns, len(counts_table_files))
    n_row = max(int(len(counts_table_files) / n_col), 1)
    
    # Remove noisy spots and genes (Spots are rows and genes are columns)
    counts_filtered = filter_data(counts, num_exp_genes, 
                                  num_exp_spots, min_gene_expression)
    
    has_clusters = False
    if clusters_file and os.path.isfile(clusters_file):
        clusters = pd.read_csv(clusters_file, sep="\t", header=None,
                               index_col=0, engine='c', low_memory=True)
        clusters = clusters.reindex(np.intersect1d(counts_filtered.index, clusters.index))
        if clusters.shape[0] == 0 or clusters.isna().values.any():
            sys.stderr.write("Error, cluster file does not match the input data\n")
            sys.exit(1)
        has_clusters = True
    elif clusters_file:
        sys.stderr.write("Error, {} is not a valid file\n".format(clusters_file))
        sys.exit(1)
        
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
        
        
    # Gene family plots
    if gene_family and combine_genes != "None":
        families = [os.path.splitext(os.path.basename(x))[0] for x in gene_family]
        counts_families = pd.DataFrame(index=counts_normalized.index, columns=families)
        for f, name in zip(gene_family, families):
            
            with open(f, "r") as filehandler:
                genes = [x.rstrip() for x in filehandler.readlines()]
            if len(genes) == 0:
                print("Error, no genes were found in {}\n".format(f))
                continue
            
            # Filter the data with the genes in the set
            counts = counts_normalized.loc[:,np.intersect1d(genes, counts_normalized.columns)]
            if counts.shape[1] == 0:
                print("Error, none of the genes from {} were found in the data\n".format(f))
                continue
            genes_in_set = counts.columns.tolist()
            
            # Compute the combined score
            if combine_genes in "NaiveMean":
                present_genes = (counts > 0).sum(axis=1) / len(genes_in_set)
                counts_families.loc[:,name] = (counts.mean(axis=1) * present_genes).values
            elif combine_genes in "NaiveSum":
                present_genes = (counts > 0).sum(axis=1) / len(genes_in_set)
                counts_families.loc[:,name] = (counts.sum(axis=1) * present_genes).values    
            else:
                # For the CumSum we need to use all the genes so in order to compute p-values
                counts_families.loc[:,name] = enrichment_score(counts_normalized, genes_in_set)
                
            # Plot the data
            plotting_data = compute_plotting_data(counts_families.loc[:,name], 
                                                  families, 
                                                  0.0,
                                                  1.0,
                                                  use_global_scale)
            if len(plotting_data) == 0:
                sys.stderr.write("Error, plotting data is empty!\n")
                sys.exit(1)  
            fig, ax, sc = plot_data(plotting_data, n_col, n_row, dot_size, data_alpha, color_scale,
                                    xlim, ylim, not disable_invert_y_axes, not disable_color_bar)
            # Save the plot
            fig.suptitle(name, fontsize=16)
            fig.savefig(os.path.join(outdir,"Combined_{}_joint_plot.pdf".format(name)), 
                        format='pdf', dpi=90)
            plt.close(fig)
            
        # Save the proportions
        counts_families.to_csv("gene_families.tsv", sep="\t")
                
    # Gene plots
    if filter_genes:
        try:
            counts_final = filter_data_genes(counts_normalized, filter_genes)
            # Compute plotting data and plot genes
            for gene in counts_final.columns:
                print("Plotting gene {}".format(gene))
                plotting_data = compute_plotting_data(counts_final.loc[:,gene], 
                                                      names, 
                                                      cutoff,
                                                      cutoff_upper,
                                                      use_global_scale)
                if len(plotting_data) == 0:
                    sys.stderr.write("Error, plotting data is empty!\n")
                    sys.exit(1)  
                fig, ax, sc = plot_data(plotting_data, n_col, n_row, dot_size, data_alpha, color_scale,
                                        xlim, ylim, not disable_invert_y_axes, not disable_color_bar)
                # Save the plot
                fig.suptitle(gene, fontsize=16)
                fig.savefig(os.path.join(outdir,"{}_joint_plot.pdf".format(gene)), format='pdf', dpi=90)
                plt.close(fig)
        except RuntimeError as e:
            sys.stdount.write("No genes could be found in the data...\n")

    if has_clusters:
        # Compute data for clusters and plot
        plotting_data = compute_plotting_data_clusters(counts_normalized, names, clusters)
        if len(plotting_data) == 0:
            sys.stderr.write("Error, plotting data is empty!\n")
            sys.exit(1)  
        fig, ax, sc = plot_data(plotting_data, n_col, n_row, dot_size, data_alpha, color_scale_clusters,
                                xlim, ylim, not disable_invert_y_axes, not disable_color_bar)
        # Save the plot
        fig.suptitle("Clusters", fontsize=16)
        fig.savefig(os.path.join(outdir, "Clusters_joint_plot.pdf"), format='pdf', dpi=90)
        plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--counts-files", required=True, nargs='+', type=str,
                        help="One or more matrices of counts (spots as rows and genes as columns)")
    parser.add_argument("--num-exp-genes", default=0.0, metavar="[FLOAT]", type=float,
                        help="The percentage of number of expressed genes (>= --min-gene-expression) a spot\n" \
                        "must have to be kept from the distribution of all expressed genes (0.0 - 1.0) (default: %(default)s)")
    parser.add_argument("--num-exp-spots", default=0.0, metavar="[FLOAT]", type=float,
                        help="The percentage of number of expressed spots a gene\n" \
                        "must have to be kept from the total number of spots (0.0 - 1.0) (default: %(default)s)")
    parser.add_argument("--min-gene-expression", default=1, type=float, metavar="[FLOAT]",
                        help="The minimum count (number of reads) a gene must have in a spot to be\n"
                        "considered expressed (default: %(default)s)")
    parser.add_argument("--cutoff", default=0.0, metavar="[FLOAT]", type=float,
                        help="The percentage of reads a gene must have in a spot to be included in the plots from\n" \
                        "the distribution of reads of the gene across all the spots (0.0 - 1.0) (default: %(default)s)")
    parser.add_argument("--cutoff-upper", default=1.0, metavar="[FLOAT]", type=float,
                        help="The percentage of reads a gene should not have in a spot to be included in the plots from\n" \
                        "the distribution of reads of the gene across all the spots (0.0 - 1.0) (default: %(default)s)")
    parser.add_argument("--data-alpha", type=float, default=1.0, metavar="[FLOAT]",
                        help="The transparency level for the data points, 0 min and 1 max (default: %(default)s)")
    parser.add_argument("--dot-size", type=int, default=20, metavar="[INT]",
                        help="The size of the data points (default: %(default)s)")
    parser.add_argument("--color-scale", default="YlOrRd", 
                        type=str, 
                        choices=["hot", "binary", "hsv", "Greys", "inferno", "YlOrRd", "bwr", "Spectral", "coolwarm"],
                        help="Different color scales for the gene plots (default: %(default)s)")
    parser.add_argument("--color-scale-clusters", default="tab20", 
                        type=str, 
                        choices=["tab20", "tab20b", "tab20c" "Set3", "Paired"],
                        help="Different color scales for the cluster plots (default: %(default)s)")
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
    parser.add_argument("--show-genes", help="Regular expression for gene symbols to be shown (one image per gene).\n" \
                        "The genes matching the reg-exp will be shown in separate files.",
                        required=False,
                        default=None,
                        type=str,
                        nargs='+')
    parser.add_argument("--clusters", help="Path to a tab delimited file containing clustering results for each spot.\n" \
                        "First column spot id and second column the cluster number (integer).",
                        default=None,
                        type=str)
    parser.add_argument("--gene-family", help="Path to one or more files containing set of genes (one per row).\n" \
                        "A combined image will be generated using the value of --combine-genes",
                        required=False,
                        default=None,
                        type=str,
                        nargs='+')
    parser.add_argument("--outdir", default=None, help="Path to output dir")
    parser.add_argument("--use-log-scale", action="store_true", default=False, 
                        help="Plot expression in log space (log2)")
    parser.add_argument("--use-global-scale", action="store_true", default=False, 
                        help="Use a global color scale instead of a relative color scale")
    parser.add_argument("--num-columns", default=1, type=int, metavar="[INT]",
                        help="The number of columns (default: %(default)s)")
    parser.add_argument("--xlim", default=[1,33], nargs='+', metavar="[FLOAT]", type=float,
                        help="The x axis limits to have equally sized sub-images (default: %(default)s)")
    parser.add_argument("--ylim", default=[1,35], nargs='+', metavar="[FLOAT]", type=float,
                        help="The y axis limits to have equally sized sub-images (default: %(default)s)")
    parser.add_argument("--disable-invert-y-axes", action="store_true", default=False,
                        help="Whether to disable the invert of the y axes or not (default False)")
    parser.add_argument("--disable-color-bar", action="store_true", default=False,
                        help="Whether to disable the color bar or not (default False)")
    parser.add_argument("--combine-genes", default="None", 
                        type=str, 
                        choices=["None", "NaiveMean", "NaiveSum", "CumSum"],
                        help="Whether to generate a combined plot with the all the genes given in --show-genes:\n" \
                        "None = do not create combined plot\n" \
                        "NaiveMean = create combine plot using the mean value of the genes in the spot adjusted by size\n" \
                        "NaiveSum = create combine plot using the sum value of the genes in the spot adjusted by size\n" \
                        "CumSum = create combined plot using a cumulative sum of the genes (0.90) and the Fisher test\n" \
                        "(default: %(default)s)")
    args = parser.parse_args()

    main(args.counts_files,
         args.cutoff,
         args.cutoff_upper,
         args.data_alpha,
         args.dot_size,
         args.normalization,
         args.color_scale,
         args.color_scale_clusters,
         args.show_genes,
         args.clusters,
         args.gene_family,
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
         args.disable_invert_y_axes,
         args.disable_color_bar,
         args.combine_genes)
