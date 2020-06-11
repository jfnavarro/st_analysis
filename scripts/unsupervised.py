#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This tool performs unsupervised learning on one or more
Spatial Transcriptomics datasets (matrices of counts)
It takes a list of datasets as input and outputs (for each given input):

 - a file containing the reduced coordinates and their labels
 - a scatter plot (dimensionality reduction space) colored by computed classes

The input data frames must have the gene names as columns and the spots coordinates as rows.

The user can select what clustering algorithm to use, what 
dimensionality reduction technique to use and normalization method to use. 

Noisy spots (very few genes expressed) are removed using a parameter.
Noisy genes (expressed in very few spots) are removed using a parameter.

@Author Jose Fernandez Navarro <jc.fernandez.navarro@gmail.com>
"""
import argparse
import sys
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FastICA, SparsePCA, FactorAnalysis
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import umap
from stanalysis.visualization import scatter_plot, scatter_plot3d, histogram
from stanalysis.preprocessing import *
from stanalysis.analysis import linear_conv
from collections import defaultdict
import matplotlib.pyplot as plt

def main(counts_table_files, 
         normalization, 
         num_clusters,
         num_exp_genes,
         num_exp_spots,
         min_gene_expression,
         num_genes_discard,
         clustering, 
         dimensionality, 
         use_log_scale, 
         num_dimensions, 
         spot_size,
         top_genes_criteria,
         outdir,
         tsne_perplexity,
         tsne_theta,
         umap_neighbors,
         umap_min_dist,
         umap_metric,
         tsne_initial_dims,
         pca_auto_components,
         dbscan_min_size,
         dbscan_eps,
         SEED):

    if len(counts_table_files) == 0 or \
    any([not os.path.isfile(f) for f in counts_table_files]):
        sys.stderr.write("Error, input file/s not present or invalid format\n")
        sys.exit(1)         
         
    if num_clusters is None and clustering != "DBSCAN":
        sys.stderr.write("Error, num_clusters must be given if clustering algorith is not DBSCAN\n")
        sys.exit(1) 
      
    if tsne_theta < 0.0 or tsne_theta > 1.0:
        sys.stdout.write("Warning, invalid value for theta. Using default..\n")
        tsne_theta = 0.5
                 
    if num_exp_genes < 0 or num_exp_spots < 0:
        sys.stdout.write("Error, min_exp_genes and min_exp_spots must be >= 0.\n")
        sys.exit(1) 
        
    if tsne_initial_dims <= num_dimensions and clustering == "tSNE":
        sys.stdout.write("Error, number of initial dimensions cannot be <= than the number of dimensions.\n")
        sys.exit(1)
        
    if pca_auto_components is not None and (pca_auto_components <= 0.0 or pca_auto_components > 1.0):
        sys.stdout.write("Error, pca_auto_components must be > 0 and <= 1.0.\n")
        sys.exit(1)
                
    if dbscan_eps <= 0.0:
        sys.stdout.write("Warning, invalid value for DBSCAN eps. Using default..\n")
        dbscan_eps = 0.5
                 
    if num_exp_genes < 0 or num_exp_genes > 1:
        sys.stderr.write("Error, invalid number of expressed genes \n")
        sys.exit(1)
         
    if num_exp_spots < 0 or num_exp_spots > 1:
        sys.stderr.write("Error, invalid number of expressed genes \n")
        sys.exit(1)
        
    if num_genes_discard < 0 or num_genes_discard > 1:
        sys.stderr.write("Error, invalid number of genes to discard \n")
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
    counts = remove_noise(counts, 
                          num_exp_genes, 
                          num_exp_spots, 
                          min_expression=min_gene_expression)

    if len(counts.index) < 5 or len(counts.columns) < 5:
        sys.stdout.write("Error, too many spots/genes were filtered.\n")
        sys.exit(1) 
                
    # Normalize data
    print("Computing per spot normalization...")
    norm_counts = normalize_data(counts, 
                                 normalization, 
                                 center=False)

    if use_log_scale:
        print("Using pseudo-log counts log1p(counts)")
        norm_counts = np.log1p(norm_counts)  

    # Keep top genes (variance or expressed)
    norm_counts = keep_top_genes(norm_counts, 
                                 num_genes_discard, 
                                 criteria=top_genes_criteria)
    
    if "None" not in dimensionality:
        print("Performing dimensionality reduction...") 
        
    if "tSNE" in dimensionality:
        # First PCA and then TSNE
        if norm_counts.shape[1] > tsne_initial_dims:
            y = PCA(whiten=True, 
                    n_components=tsne_initial_dims).fit_transform(norm_counts)
        else:
            y = norm_counts
        local_perplexity = min(y.shape[0] / 3.5, tsne_perplexity)
        reduced_data = TSNE(n_components=num_dimensions,
                            angle=tsne_theta, 
                            random_state=SEED,
                            perplexity=tsne_perplexity).fit_transform(y)
    elif "PCA" in dimensionality:
        n_comps = num_dimensions
        solver = "auto"
        if pca_auto_components is not None:
            n_comps = pca_auto_components
            solver = "full"
        reduced_data = PCA(n_components=n_comps, 
                           svd_solver=solver, 
                           whiten=True,
                           random_state=SEED,
                           copy=True).fit_transform(norm_counts)
    elif "ICA" in dimensionality:
        reduced_data = FastICA(n_components=num_dimensions, 
                               algorithm='parallel', 
                               whiten=True,
                               fun='logcosh', 
                               w_init=None, 
                               random_state=SEED).fit_transform(norm_counts)
    elif "SPCA" in dimensionality:
        import multiprocessing
        reduced_data = SparsePCA(n_components=num_dimensions, 
                                 alpha=1, 
                                 random_state=SEED,
                                 n_jobs=multiprocessing.cpu_count()-1)
    elif "FactorAnalysis" in dimensionality:
        reduced_data = FactorAnalysis(n_components=num_dimensions,
                                      random_state=SEED).fit_transform(norm_counts)
    else: 
        reduced_data = umap.UMAP(n_neighbors=umap_neighbors,
                                 min_dist=umap_min_dist,
                                 n_components=num_dimensions,
                                 random_state=SEED,
                                 metric=umap_metric).fit_transform(norm_counts)
        
    print("Performing clustering...")
    # Do clustering on the dimensionality reduced coordinates
    if "KMeans" in clustering:
        labels = KMeans(init='k-means++',
                        n_clusters=num_clusters,
                        n_init=10).fit_predict(reduced_data)
    elif "Hierarchical" in clustering:
        labels = AgglomerativeClustering(n_clusters=num_clusters,
                                         affinity='euclidean',
                                         linkage='ward').fit_predict(reduced_data)
    elif "DBSCAN" in clustering:
        labels = DBSCAN(eps=dbscan_eps, 
                        min_samples=dbscan_min_size, 
                        metric='euclidean', 
                        n_jobs=-1).fit_predict(reduced_data)
    else:
        gm = GaussianMixture(n_components=num_clusters,
                             covariance_type='full').fit(reduced_data)
        labels = gm.predict(reduced_data)
        
    # Check if there are -1 in the labels and that the number of labels is correct
    if -1 in labels or len(labels) != len(norm_counts.index):
        sys.stderr.write("Error, something went wrong in the clustering..\n")
        sys.exit(1)
        
    # If cluster 0 sum 1
    if 0 in labels:
        labels = labels + 1
        
    # Plot the clustered spots with the class color in the reduced space
    if num_dimensions == 3:
        scatter_plot3d(x_points=reduced_data[:,0], 
                       y_points=reduced_data[:,1],
                       z_points=reduced_data[:,2],
                       colors=labels, 
                       output=os.path.join(outdir, "computed_dim_red_3D.pdf"),
                       title='Computed classes (color)', 
                       alpha=0.8,
                       size=spot_size)
        with open(os.path.join(outdir, "computed_dim_red_3D.tsv"), "w") as filehandler:
            for s,x,y,z,l in zip(norm_counts.index,
                               reduced_data[:,0], 
                               reduced_data[:,1], 
                               reduced_data[:,2], 
                               labels):
                filehandler.write("{}\t{}\t{}\t{}\t{}\n".format(s,x,y,z,l))   
    else:
        scatter_plot(x_points=reduced_data[:,0], 
                     y_points=reduced_data[:,1],
                     colors=labels, 
                     output=os.path.join(outdir, "computed_dim_red_2D.pdf"),
                     title='Computed classes (color)', 
                     alpha=0.8,
                     invert_y=False,
                     size=spot_size)
        with open(os.path.join(outdir, "computed_dim_red_2D.tsv"), "w") as filehandler:
            for s,x,y,l in zip(norm_counts.index,
                               reduced_data[:,0], 
                               reduced_data[:,1], 
                               labels):
                filehandler.write("{}\t{}\t{}\t{}\n".format(s,x,y,l))          
                                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--counts-files", required=True, nargs='+', type=str,
                        help="One or more matrices with gene counts per spot (genes as columns)")
    parser.add_argument("--normalization", default="RAW",
                        type=str, 
                        choices=["RAW", "REL", "CPM"],
                        help="Normalize the counts using:\n" \
                        "RAW = absolute counts\n" \
                        "REL = Each gene count divided by the total count of its spot\n"
                        "CPM = Each gene count divided by the total count of its spot multiplied by its mean\n"
                        "(default: %(default)s)")
    parser.add_argument("--num-clusters", default=None, metavar="[INT]", type=int, choices=range(2, 30),
                        help="The number of clusters/regions expected to find.\n" \
                        "Note that this parameter has no effect with DBSCAN clustering.")
    parser.add_argument("--num-exp-genes", default=0.01, metavar="[FLOAT]", type=float,
                        help="The percentage of number of expressed genes (>= --min-gene-expression) a spot\n"
                        "must have to be kept from the distribution of all expressed genes (0.0 - 1.0) (default: %(default)s)")
    parser.add_argument("--num-exp-spots", default=0.01, metavar="[FLOAT]", type=float,
                        help="The percentage of number of expressed spots a gene\n"
                        "must have to be kept from the total number of spots (0.0 - 1.0) (default: %(default)s)")
    parser.add_argument("--min-gene-expression", default=1, type=int, metavar="[INT]", choices=range(1, 50),
                        help="The minimum count (number of reads) a gene must have in a spot to be\n"
                        "considered expressed (default: %(default)s)")
    parser.add_argument("--num-genes-discard", default=0.5, metavar="[FLOAT]", type=float,
                        help="The percentage of genes (0.0 - 1.0) to discard from the distribution of all the genes\n"
                        "across all the spots using the variance or the top highest expressed\n"
                        "(see --top-genes-criteria)\n " \
                        "Low variance or lowly expressed will be discarded (default: %(default)s)")
    parser.add_argument("--clustering", default="KMeans",
                        type=str, choices=["Hierarchical", "KMeans", "DBSCAN", "Gaussian"],
                        help="What clustering algorithm to use after the dimensionality reduction:\n"
                        "Hierarchical = Hierarchical clustering (Ward)\n"
                        "KMeans = Suitable for small number of clusters\n"
                        "DBSCAN = Number of clusters will be automatically inferred\n"
                        "Gaussian = Gaussian Mixtures Model\n"
                        "(default: %(default)s)")
    parser.add_argument("--dimensionality", default="tSNE",
                        type=str, choices=["None", "tSNE", "PCA", "ICA", "SPCA", "FactorAnalysis", "UMAP"],
                        help="What dimensionality reduction algorithm to use before the clustering:\n"
                        "None = no dimensionality reduction\n"
                        "tSNE = t-distributed stochastic neighbor embedding\n"
                        "PCA = Principal component analysis\n"
                        "ICA = Independent component analysis\n"
                        "SPCA = Sparse principal component analysis\n"
                        "FactorAnalysis = Linear model with Gaussian latent variables\n"
                        "UMAP = Uniform Manifold Approximation and Projection\n"
                        "(default: %(default)s)")
    parser.add_argument("--use-log-scale", action="store_true", default=False,
                        help="Transform the counts to log2(counts + 1) after normalization")
    parser.add_argument("--num-dimensions", default=2, metavar="[INT]", type=int,
                        help="The number of dimensions to use in the dimensionality reduction. (default: %(default)s)")
    parser.add_argument("--spot-size", default=4, metavar="[INT]", type=int,
                        help="The size of the spots when generating the plots. (default: %(default)s)")
    parser.add_argument("--top-genes-criteria", default="Variance", metavar="[STR]", 
                        type=str, choices=["Variance", "TopRanked"],
                        help="What criteria to use to keep top genes before doing\n"
                        "the dimensionality reduction (Variance or TopRanked) (default: %(default)s)")
    parser.add_argument("--tsne-perplexity", default=30, metavar="[INT]", type=int,
                        help="The value of the perplexity for the t-SNE method. (default: %(default)s)")
    parser.add_argument("--tsne-theta", default=0.5, metavar="[FLOAT]", type=float,
                        help="The value of theta for the t-SNE method. (default: %(default)s)")
    parser.add_argument("--umap-neighbors", default=15, metavar="[INT]", type=int,
                        help="The number of neighboring points used in local approximations of manifold structure (UMAP) (default: %(default)s)")
    parser.add_argument("--umap-min-dist", default=0.1, metavar="[FLOAT]", type=float,
                        help="This controls how tightly the embedding is allowed to compress points together (UMAP) (default: %(default)s)")
    parser.add_argument("--umap-metric", default="euclidean", metavar="[STR]", type=str,
                        help="This controls how the distance is computed in the ambient space of the input data (UMAP) (default: %(default)s)")
    parser.add_argument("--tsne-initial-dims", default=50, metavar="[INT]", type=int,
                        help="The number of initial dimensions of the PCA step in the t-SNE clustering. (default: %(default)s)")
    parser.add_argument("--outdir", default=None, help="Path to output dir")
    parser.add_argument("--pca-auto-components", default=None, metavar="[FLOAT]", type=float,
                        help="For the PCA dimensionality reduction the number of dimensions\n"
                        "are computed so to include the percentage of variance given as input [0.1-1].")
    parser.add_argument("--dbscan-min-size", default=5, metavar="[INT]", type=int,
                        help="The value of the minimum cluster sizer for DBSCAN. (default: %(default)s)")
    parser.add_argument("--dbscan-eps", default=0.5, metavar="[FLOAT]", type=float,
                        help="The value of the EPS parameter for DBSCAN. (default: %(default)s)")
    parser.add_argument("--seed", default=999, metavar="[INT]", type=int,
                        help="The value of the random seed. (default: %(default)s)")
    args = parser.parse_args()
    main(args.counts_files, 
         args.normalization, 
         args.num_clusters,
         args.num_exp_genes,
         args.num_exp_spots,
         args.min_gene_expression,
         args.num_genes_discard,
         args.clustering, 
         args.dimensionality, 
         args.use_log_scale, 
         args.num_dimensions, 
         args.spot_size,
         args.top_genes_criteria,
         args.outdir,
         args.tsne_perplexity,
         args.tsne_theta,
         args.umap_neighbors,
         args.umap_min_dist,
         args.umap_metric,
         args.tsne_initial_dims,
         args.pca_auto_components,
         args.dbscan_min_size,
         args.dbscan_eps,
         args.seed)

