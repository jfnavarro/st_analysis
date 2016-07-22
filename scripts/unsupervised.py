#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
A script that does unsupervised
classification on single cell data (Mainly used for Spatial Transcriptomics)
It takes a list of data frames as input and outputs :

 - the normalized/filtered counts as matrix (one for each dataset)
 - a scatter plot with the predicted classes for each spot 
 - a file with the predicted classes for each spot and the spot coordinates (one for each dataset)

The spots in the output file will have the index of the dataset
appended. For instance if two datasets are given the indexes will
be (1 and 2). 

The user can select what clustering algorithm to use
and what dimensionality reduction technique to use. 

Noisy spots (very few genes expressed) are removed using a parameter.
Noisy genes (expressed in very few spots) are removed using a parameter.

The user can optionally give a list of images
and image alignments to plot the predicted classes
on top of the image. Then one image for each dataset
will be generated.

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
from stanalysis.visualization import scatter_plot
from stanalysis.normalization import computeSizeFactors
from stanalysis.alignment import parseAlignmentMatrix

DIMENSIONS = 2

def main(counts_table_files, 
         normalization, 
         num_clusters, 
         clustering_algorithm, 
         dimensionality_algorithm,
         use_log_scale,
         num_exp_genes, 
         num_genes_keep,
         outdir,
         alignment_files, 
         image_files):

    if len(counts_table_files) == 0 or any([not os.path.isfile(f) for f in counts_table_files]):
        sys.stderr.write("Error, input file/s not present or invalid format\n")
        sys.exit(1)
            
    if outdir is None: 
        outdir = os.getcwd()
       
    num_exp_genes = num_exp_genes / 100.0
    num_genes_keep = num_genes_keep / 100.0
    
    # Spots are rows and genes are columns
    # Merge all the datasets into one and append the dataset name to each row
    index_to_spots = [[] for ele in xrange(len(counts_table_files))]
    counts = pd.DataFrame()
    for i,counts_file in enumerate(counts_table_files):
        new_counts = pd.read_table(counts_file, sep="\t", header=0, index_col=0)
        new_spots = ["{0}_{1}".format(i, spot) for spot in new_counts.index]
        new_counts.index = new_spots
        counts = counts.append(new_counts)
        index_to_spots[i].append(new_spots)
    counts.fillna(0.0, inplace=True)
    
    # How many spots do we keep based on the number of genes expressed?
    min_genes_spot_exp = round((counts != 0).sum(axis=1).quantile(num_exp_genes))
    print "Number of expressed genes a spot must have to be kept " \
    "({0}% of total expressed genes) {1}".format(num_exp_genes, min_genes_spot_exp)
    # Remove noisy spots  
    counts = counts[(counts != 0).sum(axis=1) >= min_genes_spot_exp]
    
    # Spots are columns and genes are rows
    counts = counts.transpose()
    
    # Normalization
    if normalization in "DESeq":
        size_factors = computeSizeFactors(counts, function=np.median)
        norm_counts = counts.div(size_factors) 
    elif normalization in "REL":
        spots_sum = counts.sum(axis=1)
        norm_counts = counts.div(spots_sum) 
    elif normalization in "RAW":
        norm_counts = counts
    else:
        sys.stderr.write("Error, incorrect normalization method\n")
        sys.exit(1)
        
    # This could be another normalization option
    # Scale spots (columns) against the mean and variance
    #norm_counts = pd.DataFrame(data=scale(norm_counts, axis=1, with_mean=True, with_std=True), 
    #                           index=norm_counts.index, columns=norm_counts.columns)
    
    # Keep only the genes with higher over-all expression
    # NOTE: this could be changed so to keep the genes with the highest variance
    min_genes_spot_var = norm_counts.sum(axis=1).quantile(num_genes_keep)
    print "Min normalized expression a gene must have over all spot " \
    "to be kept ({0}% of total) {1}".format(num_genes_keep, min_genes_spot_var)
    norm_counts = norm_counts[norm_counts.sum(axis=1) >= min_genes_spot_var]
    
    # Spots as rows and genes as columns
    norm_counts = norm_counts.transpose()
        
    # Write normalized counts to different files
    tot_spots = norm_counts.index
    for i in xrange(len(counts_table_files)):
        spots_to_keep = [spot for spot in tot_spots if spot.startswith("{}_".format(i))]
        slice_counts = norm_counts.loc[spots_to_keep]
        slice_counts.index = [spot.split("_")[1] for spot in spots_to_keep]
        slice_counts.to_csv(os.path.join(outdir, "normalized_counts_{}.tsv".format(i)), sep="\t")
              
    if "tSNE" in dimensionality_algorithm:
        # method = barnes_hut or exact(slower)
        # init = pca or random
        # random_state = None or number
        # metric = euclidean or any other
        # n_components = 2 is default
        decomp_model = TSNE(n_components=DIMENSIONS, random_state=None, perplexity=5,
                            early_exaggeration=4.0, learning_rate=1000, n_iter=1000,
                            n_iter_without_progress=30, metric="euclidean", init="pca",
                            method="barnes_hut", angle=0.0)
    elif "PCA" in dimensionality_algorithm:
        # n_components = None, number of mle to estimate optimal
        decomp_model = PCA(n_components=DIMENSIONS, whiten=True, copy=True)
    elif "ICA" in dimensionality_algorithm:
        decomp_model = FastICA(n_components=DIMENSIONS, 
                               algorithm='parallel', whiten=True,
                               fun='logcosh', w_init=None, random_state=None)
    elif "SPCA" in dimensionality_algorithm:
        decomp_model = SparsePCA(n_components=DIMENSIONS, alpha=1)
    else:
        sys.stderr.write("Error, incorrect dimensionality reduction method\n")
        sys.exit(1)
    
    
    if use_log_scale:
        print "Using log counts"
        norm_counts = np.log2(norm_counts + 1)    
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

    # Obtain predicted classes for each spot
    labels = clustering.fit_predict(reduced_data)
    if 0 in labels: labels = labels + 1
    
    # Plot the clustered spots with the class color
    scatter_plot(x_points=reduced_data[:,0], 
                 y_points=reduced_data[:,1], 
                 colors=labels, 
                 output=os.path.join(outdir,"computed_classes.png"), 
                 alignment=None, 
                 cmap=None, 
                 title='Computed classes', 
                 xlabel='X', 
                 ylabel='Y',
                 image=None, 
                 alpha=1.0, 
                 size=50)
    
    # Write the spots and their classes to a file
    assert(len(labels) == len(norm_counts.index))
    # First get the spots coordinates
    x_points_index = [[] for ele in xrange(len(counts_table_files))]
    y_points_index = [[] for ele in xrange(len(counts_table_files))]
    labels_index = [[] for ele in xrange(len(counts_table_files))]
    file_writers = [open("computed_classes_{}.txt".format(i),"w") for i in xrange(len(counts_table_files))]
    # Write the coordinates and the label/class the belong to
    for i,bc in enumerate(norm_counts.index):
        # bc is i_XxY
        tokens = bc.split("x")
        assert(len(tokens) == 2)
        y = float(tokens[1])
        x = float(tokens[0].split("_")[1])
        index = int(tokens[0].split("_")[0])
        x_points_index[index].append(x)
        y_points_index[index].append(y)
        labels_index[index].append(labels[i])
        file_writers[index].write("{0}\t{1}\n".format(labels[i], "{}x{}".format(x,y)))
        
    # Close the files
    for file_descriptor in file_writers:
        file_descriptor.close()
        
    # Create one image for each dataset
    for i,image in enumerate(image_files) if image_files else []:
        if image is not None and os.path.isfile(image):
            alignment_file = alignment_files[i] \
            if alignment_files is not None and len(alignment_files) >= i else None
            # alignment_matrix will be identity if alignment file is None
            alignment_matrix = parseAlignmentMatrix(alignment_file)            
            scatter_plot(x_points=x_points_index[i], 
                         y_points=y_points_index[i], 
                         colors=labels_index[i], 
                         output=os.path.join(outdir,"computed_classes_tissue_{}.png".format(i)), 
                         alignment=alignment_matrix, 
                         cmap=None, 
                         title='Computed classes tissue', 
                         xlabel='X', 
                         ylabel='Y',
                         image=image, 
                         alpha=1.0, 
                         size=50)
             
                                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--counts-table-files", required=True, nargs='+', type=str,
                        help="One or more matrices with gene counts per feature/spot (genes as columns)")
    parser.add_argument("--normalization", default="DESeq", metavar="[STR]", 
                        type=str, choices=["RAW", "DESeq", "REL"],
                        help="Normalize the counts using RAW(absolute counts) , " \
                        "DESeq or REL(relative counts) (default: %(default)s)")
    parser.add_argument("--num-clusters", default=3, metavar="[INT]", type=int, choices=range(2, 10),
                        help="The number of clusters/regions expected to be found. (default: %(default)s)")
    parser.add_argument("--num-exp-genes", default=2, metavar="[INT]", type=int, choices=range(0, 100),
                        help="The percentage of number of expressed genes ( != 0 ) a spot " \
                        "must have to be kept from the distribution of all expressed genes (default: %(default)s)")
    parser.add_argument("--num-genes-keep", default=3, metavar="[INT]", type=int, choices=range(0, 100),
                        help="The percentage of top expressed genes to keep from the expression distribution of all the genes " \
                        "accross all the spots (default: %(default)s)")
    parser.add_argument("--clustering-algorithm", default="KMeans", metavar="[STR]", 
                        type=str, choices=["Hierarchical", "KMeans"],
                        help="What clustering algorithm to use after the dimensionality reduction " \
                        "(Hierarchical - KMeans) (default: %(default)s)")
    parser.add_argument("--dimensionality-algorithm", default="ICA", metavar="[STR]", 
                        type=str, choices=["tSNE", "PCA", "ICA", "SPCA"],
                        help="What dimensionality reduction algorithm to use " \
                        "(tSNE - PCA - ICA - SPCA) (default: %(default)s)")
    parser.add_argument("--use-log-scale", action="store_true", default=False,
                        help="Use log values in the dimensionality reduction step.")
    parser.add_argument("--alignment-files", default=None, nargs='+', type=str,
                        help="One or more tab delimited files containing and alignment matrix for the images " \
                        "(array coordinates to pixel coordinates) as a 3x3 matrix in one row.\n" \
                        "Only useful is the image has extra borders, for instance not cropped to the array corners" \
                        "or if you want the keep the original image size in the plots.")
    parser.add_argument("--image-files", default=None, nargs='+', type=str,
                        help="When given the data will plotted on top of the image, " \
                        "It can be one ore more, ideally one for each input dataset\n " \
                        "It desirable that the image is cropped to the array corners otherwise an alignment file is needed")
    parser.add_argument("--outdir", default=None, help="Path to output dir")
    args = parser.parse_args()
    main(args.counts_table_files, args.normalization, int(args.num_clusters), 
         args.clustering_algorithm, args.dimensionality_algorithm, args.use_log_scale,
         args.num_exp_genes, args.num_genes_keep, args.outdir, 
         args.alignment_files, args.image_files)

