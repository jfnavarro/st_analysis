#! /usr/bin/env python
"""
A script to make un-supervised
classification on single cell data.
It takes a data frame as input and outputs
the normalized counts (data frame), a scatter plot
with the predicted classes and file with the predicted
classes and the spot coordinates.
The user can select what clustering algorithm to use
and what dimensionality reduction technique to use. 

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
import matplotlib.pyplot as plt
from stanalysis.visualization import plotSpotsWithImage
from stanalysis.normalization import computeSizeFactors

MIN_GENES_SPOT_EXP = 0.1
MIN_GENES_SPOT_VAR = 0.1
MIN_FEATURES_GENE = 10
MIN_EXPRESION = 2
        
def main(counts_table, 
         normalization, 
         num_clusters, 
         clustering_algorithm, 
         dimensionality_algorithm,
         outdir,
         alignment, 
         image):

    if not os.path.isfile(counts_table):
        sys.stderr.write("Error, input file/s not present or invalid format\n")
        sys.exit(1)
        
    if outdir is None: outdir = "."
       
    # Spots are rows and genes are columns
    counts = pd.read_table(counts_table, sep="\t", header=0, index_col=0)

    # How many spots do we keep based on the number of genes expressed?
    min_genes_spot_exp = (counts != 0).sum(axis=1).quantile(MIN_GENES_SPOT_EXP)
    print "Number of expressed genes a spot must have to be kept " \
    "(1% of total expressed genes) " + str(min_genes_spot_exp)
    
    # Remove noisy spots  
    counts = counts[(counts != 0).sum(axis=1) >= min_genes_spot_exp]
    # Spots are columns and genes are rows
    counts = counts.transpose()
    # Remove noisy genes
    counts = counts[(counts >= MIN_EXPRESION).sum(axis=1) >= MIN_FEATURES_GENE]
    
    # Normalization
    if normalization in "DESeq":
        size_factors = computeSizeFactors(counts, function=np.median)
        norm_counts = counts.div(size_factors) 
    elif normalization in "TPM":
        #    feature.sums = apply(exp.values, 2, sum)
        #    norm.counts = (t(t(exp.values) / feature.sums)*1e6) + 1
        spots_sum = counts.sum(axis=1)
        norm_counts = ((counts.transpose().div(spots_sum)) * 1e6).transpose()
    elif normalization in "RAW":
        norm_counts = counts
    else:
        sys.stderr.write("Error, incorrect normalization method\n")
        sys.exit(1)
    
    # Scale spots (columns) against the mean and variance
    #norm_counts = pd.DataFrame(data=scale(norm_counts, axis=1, with_mean=True, with_std=True), 
    #                           index=norm_counts.index, columns=norm_counts.columns)
    
    # How many genes do we keep based on the variance?
    # TODO this could be done based on expression level (keep the highest for instance)
    min_genes_spot_var = norm_counts.var(axis=1).quantile(MIN_GENES_SPOT_VAR)
    print "Min variance a gene must have over all spot " \
    "to be kept (1% of total variance) " + str(min_genes_spot_var)
    norm_counts = norm_counts[norm_counts.var(axis=1) >= min_genes_spot_var]
    
    # Spots as rows and genes as columns
    norm_counts = norm_counts.transpose()
    # Write normalized and filtered counts to a file
    norm_counts.to_csv(os.path.join(outdir, "normalized_counts.txt"), sep="\t")
              
    if "tSNE" in dimensionality_algorithm:
        # method = barnes_hut or exact(slower)
        # init = pca or random
        # random_state = None or number
        # metric = euclidean or any other
        # n_components = 2 is default
        decomp_model = TSNE(n_components=2, random_state=None, perplexity=5,
                            early_exaggeration=4.0, learning_rate=1000, n_iter=1000,
                            n_iter_without_progress=30, metric="euclidean", init="pca",
                            method="barnes_hut", angle=0.5)
    elif "PCA" in dimensionality_algorithm:
        # n_components = None, number of mle to estimate optimal
        decomp_model = PCA(n_components=2, whiten=True, copy=True)
    elif "ICA" in dimensionality_algorithm:
        decomp_model = FastICA(n_components=2, 
                               algorithm='parallel', whiten=True,
                               fun='logcosh', w_init=None, random_state=None)
    elif "SPCA" in dimensionality_algorithm:
        decomp_model = SparsePCA(n_components=2, alpha=1)
    else:
        sys.stderr.write("Error, incorrect dimensionality reduction method\n")
        sys.exit(1)
    
    # Use log2 counts if we do not center the data
    reduced_data = decomp_model.fit_transform(np.log2(norm_counts + 1))
    
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

    labels = clustering.fit_predict(reduced_data)
    if 0 in labels: labels = labels + 1
    
    # Plot the clustered spots with the class color
    fig = plt.figure(figsize=(8,8))
    a = fig.add_subplot(111, aspect='equal')
    a.scatter(reduced_data[:,0], reduced_data[:,1], c=labels, s=50)
    fig.savefig(os.path.join(outdir,"computed_classes_scatter.png"))
    
    # Write the spots and their classes to a file
    assert(len(labels) == len(norm_counts.index))
    # First get the spots coordinates
    x_points = list()
    y_points = list()
    # Write the coordinates and the label/class the belong to
    with open(os.path.join(outdir, "computed_classes.txt"), "w") as filehandler:
        for i,bc in enumerate(norm_counts.index):
            # bc is XxY
            tokens = bc.split("x")
            assert(len(tokens) == 2)
            x = int(tokens[0])
            y = int(tokens[1])
            x_points.append(x)
            y_points.append(y)
            filehandler.write(str(labels[i]) + "\t" + str(x) + "\t" + str(y) + "\n")
    
    if image is not None and os.path.isfile(image):
        plotSpotsWithImage(x_points, y_points, labels, image,
                           "computed_classes_tissue.png", alignment)

                                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts-table", required=True,
                        help="A table with gene counts per feature/spot (genes as columns)")
    parser.add_argument("--normalization", default="DESeq", metavar="[STR]", 
                        type=str, choices=["RAW", "DESeq", "TPM"],
                        help="Normalize the counts using (RAW - DESeq - TPM) (default: %(default)s)")
    parser.add_argument("--num-clusters", default=3, metavar="[INT]", type=int, choices=range(2, 10),
                        help="If given the number of clusters will be adjusted. " \
                        "Otherwise they will be pre-computed (default: %(default)s)")
    parser.add_argument("--clustering-algorithm", default="KMeans", metavar="[STR]", 
                        type=str, choices=["Hierarchical", "KMeans"],
                        help="What clustering algorithm to use after the dimensionality reduction " \
                        "(Hierarchical - KMeans) (default: %(default)s)")
    parser.add_argument("--dimensionality-algorithm", default="tSNE", metavar="[STR]", 
                        type=str, choices=["tSNE", "PCA", "ICA", "SPCA"],
                        help="What dimensionality reduction algorithm to use " \
                        "(tSNE - PCA - ICA - SPCA) (default: %(default)s)")
    parser.add_argument("--alignment", default=None,
                        help="A file containing the alignment image " \
                        "(array coordinates to pixel coordinates) as a 3x3 matrix")
    parser.add_argument("--image", default=None, 
                        help="When given the data will plotted on top of the image, \
                        if the alignment matrix is given the data will be aligned")
    parser.add_argument("--outdir", default=None, help="Path to output dir")
    args = parser.parse_args()
    main(args.counts_table, args.normalization, int(args.num_clusters), 
         args.clustering_algorithm, args.dimensionality_algorithm,
         args.outdir, args.alignment, args.image)

