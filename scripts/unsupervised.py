#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
A script that does unsupervised
classification on single cell data (Mainly used for Spatial Transcriptomics)
It takes a list of data frames as input and outputs :

 - the filtered aggregated data frame
 - a scatter plot with the predicted classes for each spot 
 - a file with the predicted classes for each spot (tab delimited)
 - a scatter plot for each dataset of the spots and the predicted color 
   (if images are given, the images are used as background)

The input data frames must have the gene names as columns and
the spots coordinates as rows (1x1).

The spots in the output file will have the index of the dataset
appended. For instance if two datasets are given the indexes will
be (1 and 2). 

The user can select what clustering algorithm to use
and what dimensionality reduction technique to use. 

Noisy spots (very few genes expressed) are removed using a parameter.
Noisy genes (expressed in very few spots) are removed using a parameter.

@Author Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>
"""
import argparse
import sys
import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, FastICA, SparsePCA
#from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
#from sklearn.preprocessing import scale
from stanalysis.visualization import scatter_plot, scatter_plot3d, histogram
from stanalysis.preprocessing import *
from stanalysis.alignment import parseAlignmentMatrix
import matplotlib.pyplot as plt

def linear_conv(old, min, max, new_min, new_max):
    return ((old - min) / (max - min)) * ((new_max - new_min) + new_min)
        
def main(counts_table_files, 
         normalization, 
         num_clusters, 
         clustering_algorithm, 
         dimensionality_algorithm,
         use_log_scale,
         apply_sample_normalization,
         num_exp_genes, 
         num_genes_keep,
         outdir,
         alignment_files, 
         image_files,
         num_dimensions,
         spot_size,
         top_genes_criteria):

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
       
    num_exp_genes = num_exp_genes / 100.0
    num_genes_keep = num_genes_keep / 100.0
    
    # Merge input datasets (Spots are rows and genes are columns)
    counts = aggregate_datatasets(counts_table_files)
               
    # Per sample normalization (Spots are rows and genes are columns)
    if apply_sample_normalization and len(counts_table_files) > 1:
        print "Computing per sample normalization..."
        counts = normalize_samples(counts)

    # Remove noisy spots and genes (Spots are rows and genes are columns)
    counts = remove_noise(counts, num_exp_genes)
    
    # Write un-normalized filtered counts to a file (for downstream analysis)
    counts.to_csv(os.path.join(outdir, "aggregated_filtered_counts.tsv"), sep="\t")
    
    # Normalize data
    print "Computing per spot normalization..." 
    norm_counts = normalize_data(counts, normalization)
    
    # Keep top genes (variance or expressed)
    norm_counts = keep_top_genes(norm_counts, num_genes_keep, criteria=top_genes_criteria)
              
    if "tSNE" in dimensionality_algorithm:
        # method = barnes_hut or exact(slower)
        # init = pca or random
        # random_state = None or number
        # metric = euclidean or any other
        # n_components = 2 is default
        decomp_model = TSNE(n_components=num_dimensions, random_state=None, perplexity=30,
                            early_exaggeration=4.0, learning_rate=1000, n_iter=1000,
                            n_iter_without_progress=30, metric="euclidean", init="pca",
                            method="exact", angle=0.5, verbose=0)
    elif "PCA" in dimensionality_algorithm:
        # n_components = None, number of mle to estimate optimal
        decomp_model = PCA(n_components=num_dimensions, whiten=True, copy=True)
    elif "ICA" in dimensionality_algorithm:
        decomp_model = FastICA(n_components=num_dimensions, 
                               algorithm='parallel', whiten=True,
                               fun='logcosh', w_init=None, random_state=None)
    elif "SPCA" in dimensionality_algorithm:
        decomp_model = SparsePCA(n_components=num_dimensions, alpha=1)
    else:
        sys.stderr.write("Error, incorrect dimensionality reduction method\n")
        sys.exit(1)
    
    if use_log_scale:
        print "Using pseudo-log counts log2(counts + 1)"
        norm_counts = np.log2(norm_counts + 1)  
     
    print "Performing dimensionality reduction..."    
    # Perform dimensionality reduction, outputs a bunch of 2D coordinates
    reduced_data = decomp_model.fit_transform(norm_counts)
    
    # Do clustering of the dimensionality reduced coordinates
    if "KMeans" in clustering_algorithm:
        clustering = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)    
    elif "Hierarchical" in clustering_algorithm:
        clustering = AgglomerativeClustering(n_clusters=num_clusters, 
                                             affinity='euclidean',
                                             n_components=None, linkage='ward') 
    else:
        sys.stderr.write("Error, incorrect clustering method\n")
        sys.exit(1)

    print "Performing clustering..."  
    # Obtain predicted classes for each spot
    labels = clustering.fit_predict(reduced_data)
    if 0 in labels: labels = labels + 1
    
    # Check the number of predicted labels is correct
    assert(len(labels) == len(norm_counts.index))
    
    # Compute a color_label based on the RGB representation of the 3D dimensionality reduced
    labels_colors = list()
    x_max = max(reduced_data[:,0])
    x_min = min(reduced_data[:,0])
    y_max = max(reduced_data[:,1])
    y_min = min(reduced_data[:,1])
    x_p = reduced_data[:,0]
    y_p = reduced_data[:,1]
    z_p = None
    if num_dimensions == 3:
        z_p = reduced_data[:,2]
        z_max = max(reduced_data[:,2])
        z_min = min(reduced_data[:,2])
    for x,y,z in zip(x_p,y_p,z_p):
        r = linear_conv(x, x_min, x_max, 0.0, 1.0)
        g = linear_conv(y, y_min, y_max, 0.0, 1.0)
        b = linear_conv(z, z_min, z_max, 0.0, 1.0) if num_dimensions == 3 else 1.0
        labels_colors.append((r,g,b))

    print "Generating plots..." 
     
    # Plot the clustered spots with the class color
    if num_dimensions == 3:
        scatter_plot3d(x_points=reduced_data[:,0], 
                       y_points=reduced_data[:,1],
                       z_points=reduced_data[:,2],
                       colors=labels, 
                       output=os.path.join(outdir,"computed_classes.png"), 
                       title='Computed classes', 
                       alpha=1.0, 
                       size=100)
    else:
        scatter_plot(x_points=reduced_data[:,0], 
                     y_points=reduced_data[:,1],
                     colors=labels, 
                     output=os.path.join(outdir,"computed_classes.png"), 
                     title='Computed classes', 
                     alpha=1.0, 
                     size=80)          
    
    # Plot the spots with colors corresponding to the predicted class
    # Use the HE image as background if the image is given
    tot_spots = norm_counts.index
    with open(os.path.join(outdir,"computed_classes.txt"), "w") as filehandler:
        for i in xrange(len(counts_table_files)):
            # Get the list of spot coordinates and colors to plot for each dataset
            x_points = list()
            y_points = list()
            colors_classes = list()
            colors_dimensionality = list()
            for j,spot in enumerate(tot_spots):
                if spot.startswith("{}_".format(i)):
                    # spot is i_XxY
                    tokens = spot.split("x")
                    assert(len(tokens) == 2)
                    y = float(tokens[1])
                    x = float(tokens[0].split("_")[1])
                    x_points.append(x)
                    y_points.append(y)
                    colors_classes.append(labels[j])
                    colors_dimensionality.append(labels_colors[j])
                    # Write the predicted classes to a file (CLASS SPOT)
                    filehandler.write("{0}\t{1}\n".format(labels[j], spot))
                   
            # Retrieve alignment matrix and image if any
            image = image_files[i] if image_files is not None \
            and len(image_files) >= i else None
            alignment = alignment_files[i] if alignment_files is not None \
            and len(alignment_files) >= i else None
            
            # alignment_matrix will be identity if alignment file is None
            alignment_matrix = parseAlignmentMatrix(alignment)            
            scatter_plot(x_points=x_points, 
                         y_points=y_points,
                         colors=colors_classes,
                         output=os.path.join(outdir,"computed_classes_tissue_{}.png".format(i)), 
                         alignment=alignment_matrix, 
                         cmap=None, 
                         title='Computed classes tissue', 
                         xlabel='X', 
                         ylabel='Y',
                         image=image, 
                         alpha=1.0, 
                         size=spot_size)
            scatter_plot(x_points=x_points, 
                        y_points=y_points,
                        colors=colors_dimensionality, 
                        output=os.path.join(outdir,"dimensionality_color_tissue_{}.png".format(i)), 
                        alignment=alignment_matrix, 
                        cmap=plt.get_cmap("hsv"), 
                        title='Dimensionality color tissue', 
                        xlabel='X', 
                        ylabel='Y',
                        image=image, 
                        alpha=1.0, 
                        size=spot_size)           
                                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--counts-table-files", required=True, nargs='+', type=str,
                        help="One or more matrices with gene counts per feature/spot (genes as columns)")
    parser.add_argument("--normalization", default="DESeq", metavar="[STR]", 
                        type=str, choices=["RAW", "DESeq", "DESeq2", "DESeq2Log", "EdgeR", "REL"],
                        help="Normalize the counts using RAW(absolute counts) , " \
                        "DESeq, DESeq2, DESeq2Log, EdgeR and " \
                        "REL(relative counts, each gene count divided by the total count of its spot) (default: %(default)s)")
    parser.add_argument("--num-clusters", default=3, metavar="[INT]", type=int, choices=range(2, 10),
                        help="The number of clusters/regions expected to be found. (default: %(default)s)")
    parser.add_argument("--num-exp-genes", default=10, metavar="[INT]", type=int, choices=range(0, 100),
                        help="The percentage of number of expressed genes ( != 0 ) a spot " \
                        "must have to be kept from the distribution of all expressed genes (default: %(default)s)")
    parser.add_argument("--num-genes-keep", default=20, metavar="[INT]", type=int, choices=range(0, 100),
                        help="The percentage of top genes to discard from the distribution of all the genes " \
                        "across all the spots (default: %(default)s)")
    parser.add_argument("--clustering", default="KMeans", metavar="[STR]", 
                        type=str, choices=["Hierarchical", "KMeans"],
                        help="What clustering algorithm to use after the dimensionality reduction " \
                        "(Hierarchical - KMeans) (default: %(default)s)")
    parser.add_argument("--dimensionality", default="ICA", metavar="[STR]", 
                        type=str, choices=["tSNE", "PCA", "ICA", "SPCA"],
                        help="What dimensionality reduction algorithm to use " \
                        "(tSNE - PCA - ICA - SPCA) (default: %(default)s)")
    parser.add_argument("--use-log-scale", action="store_true", default=False,
                        help="Use log values in the dimensionality reduction step")
    parser.add_argument("--normalize-samples", action="store_true", default=False,
                        help="When multiple datasets given this option computes normalization " \
                        "factors for each gene using DESeq on the different samples")
    parser.add_argument("--alignment-files", default=None, nargs='+', type=str,
                        help="One or more tab delimited files containing and alignment matrix for the images " \
                        "(array coordinates to pixel coordinates) as a 3x3 matrix in one row.\n" \
                        "Only useful is the image has extra borders, for instance not cropped to the array corners" \
                        "or if you want the keep the original image size in the plots.")
    parser.add_argument("--image-files", default=None, nargs='+', type=str,
                        help="When given the data will plotted on top of the image, " \
                        "It can be one ore more, ideally one for each input dataset\n " \
                        "It is desirable that the image is cropped to the array " \
                        "corners otherwise an alignment file is needed")
    parser.add_argument("--num-dimensions", default=3, metavar="[INT]", type=int, choices=[2,3],
                        help="The number of dimensions to use in the dimensionality reduction (2 or 3). (default: %(default)s)")
    parser.add_argument("--spot-size", default=100, metavar="[INT]", type=int, choices=range(10, 500),
                        help="The size of the spots when generating the plots. (default: %(default)s)")
    parser.add_argument("--top-genes-criteria", default="Variance", metavar="[STR]", 
                        type=str, choices=["Variance", "TopRankded"],
                        help="What criteria to use to keep top genes before doing " \
                        "the dimensionality reduction (Variance or TopRanked) (default: %(default)s)")   
    parser.add_argument("--outdir", default=None, help="Path to output dir")
    args = parser.parse_args()
    main(args.counts_table_files, args.normalization, int(args.num_clusters), 
         args.clustering, args.dimensionality, args.use_log_scale,
         args.normalize_samples, args.num_exp_genes, args.num_genes_keep, args.outdir, 
         args.alignment_files, args.image_files, args.num_dimensions, args.spot_size,
         args.top_genes_criteria)

