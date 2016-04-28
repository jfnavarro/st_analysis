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
from deseq import DSet

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, FastICA
#from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import scale

import matplotlib.pyplot as plt
from matplotlib import transforms

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
         barcodes_file,
         alignment, 
         image):

    if outdir is None: outdir = "."
    
    if not os.path.isfile(counts_table):
        raise RuntimeError("Counts table file is not correct..")
       
    # Spots are rows and genes are columns
    counts = pd.read_table(counts_table, sep="\t", header=0)
    
    # How many spots do we keep based on the number of genes expressed?
    min_genes_spot_exp = (counts != 0).sum(axis=1).quantile(MIN_GENES_SPOT_EXP)
    print "Number of expressed genes a spot must have to be kept (1% of total expressed genes) " + str(min_genes_spot_exp)
    
    # Remove noisy spots  
    counts = counts[(counts != 0).sum(axis=1) >= min_genes_spot_exp]
    # Spots are columns and genes are rows
    counts = counts.transpose()
    # Remove noisy genes
    counts = counts[(counts >= MIN_EXPRESION).sum(axis=1) >= MIN_FEATURES_GENE]
    
    # Normalization
    if normalization in "DESeq":
        # No conditions to treat all as individual samples
        deseq_obj = DSet(counts, conds=None)
        deseq_obj.setSizeFactors(function=np.median)
        norm_counts = DSet.getNormalizedCounts(deseq_obj.data, deseq_obj.sizeFactors) 
    elif normalization in "TPM":
        #TODO finish this
        #    feature.sums = apply(exp.values, 2, sum)
        #    norm.counts = (t(t(exp.values) / feature.sums)*1e6) + 1
        norm_counts = counts + 1
    elif normalization in "RAW":
        norm_counts = counts + 1
    else:
        raise RuntimeError("Wrong normalization method..")
    
    # Scale spots (columns) against the mean and variance
    #norm_counts = pd.DataFrame(data=scale(norm_counts, axis=1, with_mean=True, with_std=True), 
    #                           index=norm_counts.index, columns=norm_counts.columns)
    
    # How many genes do we keep based on the variance?
    # TODO this could be done based on expression level (keep the highest for instance)
    min_genes_spot_var = norm_counts.var(axis=1).quantile(MIN_GENES_SPOT_VAR)
    print "Min variance a gene must have accross all spot to be kept (1% of total variance) " + str(min_genes_spot_var)
    norm_counts = norm_counts[norm_counts.var(axis=1) >= min_genes_spot_var]
    
    # Spots as rows and genes as columns
    norm_counts = norm_counts.transpose()
    # Write normalized and filtered counts to a file
    norm_counts.to_csv(os.path.join(outdir, "normalized_counts.txt"), sep="\t")
              
    if "tSNE" in dimensionality_algorithm:
        # method = barnes_hut or exact(slower)
        # init = pca or random
        # random_state = None or number
        # metric = euclidian or any other
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
    else:
        raise RuntimeError("Wrong dimensionality reduction method..")   
    
    # Use log2 counts if we do not center the data
    reduced_data = decomp_model.fit_transform(np.log2(norm_counts + 1))
    
    # Do clustering of the dimensionality reduced coordinates
    if "KMeans" in clustering_algorithm:
        clustering = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)    
    elif "Hierarchical" in clustering_algorithm:
        clustering = AgglomerativeClustering(n_clusters=num_clusters, 
                                             affinity='euclidean',
                                             n_componens=None, linkage='ward') 
    else:
        raise RuntimeError("Wrong clustering method..")  
     
    clustering.fit(reduced_data)
    labels = clustering.predict(reduced_data)
    if 0 in labels: labels = labels + 1
    
    # Plot the clustered spots with the class color
    fig = plt.figure(figsize=(8,8))
    a = fig.add_subplot(111, aspect='equal')
    a.scatter(reduced_data[:,0], reduced_data[:,1], c=labels, s=50)
    fig.savefig(os.path.join(outdir,"computed_classes.png"))
    
    # Write the spots and their classes to a file
    if barcodes_file is not None and os.path.isfile(barcodes_file):
        assert(len(labels) == len(norm_counts.index))
        # First get the barcodes coordinates
        map_barcodes = dict()
        x_points = list()
        y_points = list()
        with open(barcodes_file, "r") as filehandler:
            for line in filehandler.readlines():
                tokens = line.split()
                map_barcodes[tokens[0]] = (tokens[1],tokens[2])
        # Write barcodes, labels and coordinates
        with open(os.path.join(outdir, "computed_classes.txt"), "w") as filehandler:
            for i,bc in enumerate(norm_counts.index):
                x,y = map_barcodes[bc]
                x_points.append(int(x))
                y_points.append(int(y))
                filehandler.write(str(labels[i]) + "\t" + bc + "\t" + str(x) + "\t" + str(y) + "\n")
    
        if image is not None and os.path.isfile(image):
                # Create alignment matrix 
                alignment_matrix = np.zeros((3,3), dtype=np.float)
                alignment_matrix[0,0] = 1
                alignment_matrix[0,1] = 0
                alignment_matrix[0,2] = 0
                alignment_matrix[1,0] = 0
                alignment_matrix[1,1] = 1
                alignment_matrix[1,2] = 0
                alignment_matrix[2,0] = 0
                alignment_matrix[2,1] = 0
                alignment_matrix[2,2] = 1
                if alignment and len(alignment) == 9:
                    alignment_matrix[0,0] = alignment[0]
                    alignment_matrix[0,1] = alignment[1]
                    alignment_matrix[0,2] = alignment[2]
                    alignment_matrix[1,0] = alignment[3]
                    alignment_matrix[1,1] = alignment[4]
                    alignment_matrix[1,2] = alignment[5]
                    alignment_matrix[2,0] = alignment[6]
                    alignment_matrix[2,1] = alignment[7]
                    alignment_matrix[2,2] = alignment[8]
                # Plot spots with the color class in the tissue image
                img = plt.imread(image)
                fig = plt.figure(figsize=(8,8))
                a = fig.add_subplot(111, aspect='equal')
                base_trans = a.transData
                tr = transforms.Affine2D(matrix = alignment_matrix) + base_trans
                a.scatter(x_points, y_points, c=labels, edgecolor="none", s=50,transform=tr)   
                a.imshow(img)
                fig.set_size_inches(16, 16)
                fig.savefig(os.path.join(outdir,"computed_classes_tissue.png"), dpi=300)
                                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts-table", 
                        help="A table with gene counts per feature/spot")
    parser.add_argument("--normalization", default="DESeq",
                        help="Normalize the counts using (RAW - DESeq - TPM)")
    parser.add_argument("--num-clusters", default=3,
                        help="If given the number of clusters will be adjusted. Otherwise they will be pre-computed")
    parser.add_argument("--clustering-algorithm", default="KMeans",
                        help="What clustering algorithm to use after the dimensionality reduction (Hierarchical - KMeans)")
    parser.add_argument("--dimensionality-algorithm", default="tSNE",
                        help="What dimensionality reduction algorithm to use (tSNE - PCA - ICA)")
    parser.add_argument("--barcodes-file", default=None,
                        help="File with the barcodes and coordinates")
    parser.add_argument("--alignment", 
                        help="Alignment matrix needed when using the image", 
                        nargs="+", type=float, default=None)
    parser.add_argument("--image", default=None, 
                        help="When given the data will plotted on top of the image, \
                        if the alignment matrix is given the data will be aligned")
    parser.add_argument("--outdir", help="Path to output dir")
    args = parser.parse_args()
    main(args.counts_table, args.normalization, int(args.num_clusters), 
         args.clustering_algorithm, args.dimensionality_algorithm,
         args.outdir, args.barcodes_file, args.alignment, args.image)

