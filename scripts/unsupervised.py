#! /usr/bin/env python
#@Author Jose Fernandez
"""
A tool to make un-supervised
classification on the ST Data
"""
import argparse
import sys
import os
import numpy as np
import pandas as pd
sys.path.append('./')
import tempfile
from deseq import *
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt

def main(counts_table, normalization, num_clusters, 
         clustering_algorithm, dimensionality_algorithm, 
         features_to_keep, genes_to_keep, outdir):

    # Spots are columns and genes are rows
    counts = pd.read_table(counts_table, sep="\t", header=0).transpose()
    
    # Remove noisy spots
    counts = counts[counts[counts != 0].sum(axis=0) > 10]
    counts = counts[counts[counts > 2].sum(axis=1) > 2]
    
    if normalization in "DESeq":
        factors = counts.columns
        print factors
        deseq_obj = DSet(counts, factors)
        deseq_obj.setSizeFactors(function=np.median)
        print deseq_obj.sizeFactors
        norm_counts = DSet.getNormalizedCounts(counts, deseq_obj.sizeFactors)
    elif normalization in "TPM":
        #TODO finish this
        norm_counts = counts
    elif normalization in "RAW":
        norm_counts = counts
    else:
        raise RuntimeError("Wrong normalization method..")
    
    # Apply filter low variance
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    norm_counts = sel.fit_transform(norm_counts)
                                             
    if "tSNE" in dimensionality_algorithm:
        # method = barnest_hust or exact
        # init = pca or random
        # random_state = None or number
        # metric = euclidian or any other
        # n_components = 2 is default
        decomp_model = TSNE(n_components=10, random_state=0, perplexity=30,
                            early_exaggeration=5.0, learning_rate=1000, n_iter=1000,
                            n_iter_without_progress=30, metric="euclidean", init="pca",
                            method="exact", angle=0.5)
    elif "PCA" in dimensionality_algorithm:
        # n_components = None, number of mle to estimate optimal
        decomp_model = PCA(n_components="mle", whiten=True, copy=True)
    elif "ICA" in dimensionality_algorithm:
        decomp_model = FastICA(n_components=None, 
                               algorithm='parallel', whiten=True,
                               fun='logcosh', w_init=None, random_state=None)
    else:
        raise RuntimeError("Wrong dimensionality reduction method..")   
    
    # TODO Use log counts instead
    reduced_data = decomp_model.fit_transform(norm_counts.transpose())
        
    # For now only KMeans
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts-table", 
                        help="A table with gene counts per feature/spot")
    parser.add_argument("--normalization",
                        help="Normalize the counts using (Raw - DESeq - TPM)")
    parser.add_argument("--num-clusters", 
                        help="If given the number of clusters will be adjusted. Otherwise they will be pre-computed")
    parser.add_argument("--clustering-algorithm",
                        help="What clustering algorithm to use after the dimensionality reduction (Hierarchical - Kmeans)")
    parser.add_argument("--dimensionality-algorithm",
                        help="What dimensionality reduction algorithm to use (tSNE - PCA - ICA)")
    parser.add_argument("--features-to-keep",
                        help="Percentage of features to keep (1-100) using number of non zero genes/ctts counts per feature")
    parser.add_argument("--genes-to-keep",
                        help="Percentage of genes to keep (1-100) using number the variance across features")
    parser.add_argument("--outdir", help="Path to output dir")
    args = parser.parse_args()
    main(args.counts_table, args.normalization, args.num_clusters, 
         args.clustering_algorithm, args.dimensionality_algorithm, 
         args.features_to_keep, args.genes_to_keep, args.outdir)

