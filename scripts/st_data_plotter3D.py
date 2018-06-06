#! /usr/bin/env python
# -*- coding: utf-8 -*-
""" 
Script that creates a 3D scatter plot from an ST atlas.

It needs a matrix of counts with all the sections and a meta-data
matrix with information about the spots such as the 3D coordiantes (ML, AP and DV).

The output will be an image file for each gene given as input.

It allows to choose transparency for the data points and their size.

It allows to normalize the counts using different algorithms.

It allows to apply a cut-off on the number of reads.

@Author Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>
"""

import argparse
import plotly
from plotly.graph_objs import Scatter3d, Layout, ColorBar
from stanalysis.preprocessing import *
import pandas as pd
import numpy as np
import os
import sys

def main(counts_table,
         meta_info,
         cutoff,
         data_alpha,
         dot_size,
         normalization,
         genes,
         outdir,
         use_log_scale,
         clusters):

    if not os.path.isfile(counts_table) or not os.path.isfile(meta_info):
        sys.stderr.write("Error, input file/s not present or invalid format\n")
        sys.exit(1)
    
    if outdir is None or not os.path.isdir(outdir): 
        outdir = os.getcwd()
    outdir = os.path.abspath(outdir)
    print("Output directory {}".format(outdir))
         
    # Counts table (Spots are rows and genes are columns)
    counts = pd.read_table(counts_table, sep="\t", header=0, index_col=0)
    print("Total number of spots {}".format(len(counts.index)))
    print("Total number of genes {}".format(len(counts.columns)))

    # Meta-data table 
    meta = pd.read_table(meta_info, sep="\t", header=0, index_col=0)
    print("Total number of spots (meta data) {}".format(len(meta.index)))
    print("Total number of columns (meta data) {}".format(len(meta.columns)))
    
    # Clusters table
    valid_clusters = False
    if clusters is not None and os.path.isfile(clusters):
        clusters_dict = dict()
        with open(clusters, "r") as filehandler:
            for line in filehandler.readlines():
                tokens = line.split()
                assert(len(tokens) == 2)
                clusters_dict[tokens[0]] = tokens[1]
        valid_clusters = len(clusters_dict) > 0
            
    # Remove noisy spots and genes (Spots are rows and genes are columns)
    counts = remove_noise(counts, 1 / 100.0, 1 / 100.0, min_expression=1)
    
    # Normalization
    print("Computing per spot normalization...")
    counts = normalize_data(counts, normalization)      
    
    # Create a 3D scatter plot for each gene
    for gene in genes if genes is not None else []:
        print("Plotting gene {} ...".format(gene))
        vmin = 10e6
        vmax = -1
        x = list()
        y = list()
        z = list()
        colors = list()
        # First try the super non-optimal way
        # TODO this can be done in few lines using slicing
        # TODO no need to iterate all the spots every time
        for spot in counts.index:
            exp = counts.at[spot,gene]
            if exp > cutoff:
                x.append(float(meta.at[spot,"ML"]))
                y.append(float(meta.at[spot,"AP"]))
                z.append(float(meta.at[spot,"DV"]))
                if use_log_scale: 
                    exp = np.log2(exp)
                vmin = min(vmin, exp)
                vmax = max(vmax, exp)
                colors.append(exp)
        trace = Scatter3d(x=x, y=y, z=z,
                          mode='markers',
                          marker=dict(size=dot_size,
                                      cmin=vmin,
                                      cmax=vmax,
                                      color=colors,
                                      colorbar=ColorBar(title='Colorbar'),
                                      colorscale='Jet',
                                      opacity=data_alpha))
        if valid_clusters:
            trace2 = Scatter3d(x=x2, y=y2, z=z2,
                               mode='markers',
                               marker=dict(size=dot_size,
                                           color=clusters_colors,
                                           opacity=data_alpha))
              
        layout = Layout(margin=dict(l=0,r=0,b=0,t=0), 
                        title=gene,
                        scene=dict(xaxis=dict(title='x = Medial-lateral (mm)', range=[0, 5],),
                                   yaxis=dict(title='y = Anterior-posterior (mm)', range=[-5.9, 3],),
                                   zaxis=dict(title='z = Dorsal-ventral (mm)', range=[-7.9, 0],),))
        plotly.offline.plot({"data": [trace],"layout": layout}, filename='{}-plot.html'.format(gene))

            
    if valid_clusters:
        print("Plotting clusters...")
        x = list()
        y = list()
        z = list()
        clusters_colors = list()
        for spot in counts.index:
            if clusters_dict.has_key(spot):
                clusters_colors.append(clusters_dict[spot])
                x.append(float(meta.at[spot,"ML"]))
                y.append(float(meta.at[spot,"AP"]))
                z.append(float(meta.at[spot,"DV"]))
        trace = Scatter3d(x=x, y=y, z=z,
                          mode='markers',
                          marker=dict(size=dot_size,
                                      color=clusters_colors,
                                      opacity=data_alpha))
          
        layout = Layout(margin=dict(l=0,r=0,b=0,t=0), title="Clusters",
                        scene=dict(xaxis=dict(title='x = Medial-lateral (mm)', range=[0, 5],),
                                   yaxis=dict(title='y = Anterior-posterior (mm)', range=[-5.9, 3],),
                                   zaxis=dict(title='z = Dorsal-ventral (mm)', range=[-7.9, 0],),))
        plotly.offline.plot({"data": [trace],"layout": layout}, filename='clusters-plot.html')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--counts-table", required=True, type=str,
                        help="Matrix with gene counts per feature/spot (genes as columns)")
    parser.add_argument("--meta-info", required=True, type=str,
                        help="Matrix with the meta info registration for each spot")
    parser.add_argument("--cutoff", 
                        help="Do not include genes that falls below this reads cut off per spot (default: %(default)s)",
                        type=float, default=0.0, metavar="[FLOAT]", choices=range(0, 50))
    parser.add_argument("--data-alpha", type=float, default=1.0, metavar="[FLOAT]",
                        help="The transparency level for the data points, 0 min and 1 max (default: %(default)s)")
    parser.add_argument("--dot-size", type=int, default=5, metavar="[INT]", choices=range(1, 50),
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
                        "Scran = Deconvolution Sum Factors (Marioni et al)\n" \
                        "REL = Each gene count divided by the total count of its spot" \
                        "(default: %(default)s)")
    parser.add_argument("--show-genes", help="Gene symbols to be shown. Can be given several times.",
                        default=None,
                        type=str,
                        action='append')
    parser.add_argument("--clusters", help="Create a 3D plot of the clusters per spot given in \n" \
                        "a tab delimited file SPOT CLUSTER", default=None, type=str)
    parser.add_argument("--outdir", default=None, help="Path to output dir")
    parser.add_argument("--use-log-scale", action="store_true", default=False, help="Use log2(counts + 1) values")
    args = parser.parse_args()

    main(args.counts_table,
         args.meta_info,
         args.cutoff,
         args.data_alpha,
         args.dot_size,
         args.normalization,
         args.show_genes,
         args.outdir,
         args.use_log_scale,
         args.clusters)
