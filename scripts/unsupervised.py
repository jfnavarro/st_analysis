#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This tool performs unsupervised learning on one or more
Spatial Transcriptomics datasets (matrices of counts)
It takes a list of datasets as input and outputs (for each given input):

 - a scatter plot with the predicted classes (colored) for each spot 
 - a file containing two columns (SPOT and CLASS) for each dataset
 - a file containing the reduced coordinates and their labels
 - individual scatters plots with the spots colored by the class they belong to

The input data frames must have the gene names as columns and the spots coordinates as rows.

The user can select what clustering algorithm to use, what 
dimensionality reduction technique to use and normalization method to use. 

Noisy spots (very few genes expressed) are removed using a parameter.
Noisy genes (expressed in very few spots) are removed using a parameter.

@Author Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>
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
from stanalysis.visualization import scatter_plot, scatter_plot3d, histogram
from stanalysis.preprocessing import *
from stanalysis.alignment import parseAlignmentMatrix
from stanalysis.analysis import linear_conv
from collections import defaultdict
import matplotlib.pyplot as plt
  
def main(counts_table_files, 
         normalization, 
         num_clusters,
         num_exp_genes,
         num_exp_spots,
         min_gene_expression,
         num_genes_keep,
         clustering, 
         dimensionality, 
         use_log_scale, 
         num_dimensions, 
         spot_size,
         top_genes_criteria,
         outdir,
         tsne_perplexity,
         tsne_theta,
         tsne_initial_dims,
         pca_auto_components,
         dbscan_min_size,
         dbscan_eps,
         joint_plot,
         num_columns):

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
                 
    if num_exp_genes <= 0 or num_exp_spots <= 0:
        sys.stdout.write("Error, min_exp_genes and min_exp_spots must be > 0.\n")
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
    counts = remove_noise(counts, num_exp_genes / 100.0, num_exp_spots / 100.0, 
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
                                 num_genes_keep / 100.0, 
                                 criteria=top_genes_criteria)
      
    print("Performing dimensionality reduction...")   
    if "tSNE" in dimensionality:
        # First PCA and then TSNE
        y = PCA(n_components=tsne_initial_dims).fit_transform(norm_counts)
        reduced_data = TSNE(n_components=num_dimensions,
                            angle=tsne_theta, 
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
                           copy=True).fit_transform(norm_counts)
    elif "ICA" in dimensionality:
        reduced_data = FastICA(n_components=num_dimensions, 
                               algorithm='parallel', 
                               whiten=True,
                               fun='logcosh', 
                               w_init=None, 
                               random_state=None).fit_transform(norm_counts)
    elif "SPCA" in dimensionality:
        import multiprocessing
        reduced_data = SparsePCA(n_components=num_dimensions, 
                                 alpha=1, 
                                 n_jobs=multiprocessing.cpu_count()-1)
    elif "FactorAnalysis" in dimensionality:
        reduced_data = FactorAnalysis(n_components=num_dimensions).fit_transform(norm_counts)
    else:
        sys.stderr.write("Error, incorrect dimensionality reduction method\n")
        sys.exit(1)
    
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
    elif "Gaussian" in clustering:
        gm = GaussianMixture(n_components=num_clusters,
                             covariance_type='full').fit(reduced_data)
        labels = gm.predict(reduced_data)
    else:
        sys.stderr.write("Error, incorrect clustering method\n")
        sys.exit(1)
        
    # Check if there are -1 in the labels and that the number of labels is correct
    if -1 in labels or len(labels) != len(norm_counts.index):
        sys.stderr.write("Error, something went wrong in the clustering..\n")
        sys.exit(1)

    # Write the spots and their classes to a file
    file_writers = [open(os.path.join(outdir,
                                      "{}_clusters.tsv".format(
                                      os.path.splitext(os.path.basename(name))[0])),"w")
                    for name in counts_table_files]
    # Write the coordinates and the label/class that they belong to
    spot_plot_data = defaultdict(lambda: [[],[],[],[]])
    for i, spot in enumerate(norm_counts.index):
        tokens = spot.split("x")
        assert(len(tokens) == 2)
        y = float(tokens[1])
        tokens2 = tokens[0].split("_")
        # This is to account for the cases where the spots already contain a tag (separated by "_")
        if len(tokens2) == 3:
            x = float(tokens2[2])
        elif len(tokens2) == 2:
            x = float(tokens2[1])
        elif len(tokens2) == 1:
            x = float(tokens2[0])
        else:
            sys.stderr.write("Error, the spots in the input data have "
                             "the wrong format {}\n.".format(spot))
            sys.exit(1)
        index = int(tokens2[0]) if len(tokens2) > 1 else 0
        spot_plot_data[index][0].append(x)
        spot_plot_data[index][1].append(y)
        spot_plot_data[index][2].append(labels[i])
        # This is to account for the cases where the spots already contain a tag (separated by "_")
        if len(tokens2) == 3:
            spot_str = "{}_{}x{}".format(tokens2[1],x,y)
        else:
            spot_str = "{}x{}".format(x,y)
        file_writers[index].write("{0}\t{1}\n".format(spot_str, labels[i]))
    # Close the files
    for file_writer in file_writers:
        file_writer.close()
        
    print("Generating plots...")
    # Plot the clustered spots with the class color
    if num_dimensions == 3:
        scatter_plot3d(x_points=reduced_data[:,0], 
                       y_points=reduced_data[:,1],
                       z_points=reduced_data[:,2],
                       colors=labels, 
                       output=os.path.join(outdir,"computed_clusters.pdf"), 
                       title='Computed classes', 
                       alpha=1.0, 
                       size=20)
        with open(os.path.join(outdir,"computed_clusters_3D.tsv"), "w") as filehandler: 
            for x,y,z,l in zip(reduced_data[:,0], 
                               reduced_data[:,1], 
                               reduced_data[:,2], 
                               labels):
                filehandler.write("{}\t{}\t{}\t{}\n".format(x,y,z,l))   
    else:
        scatter_plot(x_points=reduced_data[:,0], 
                     y_points=reduced_data[:,1],
                     colors=labels, 
                     output=os.path.join(outdir,"computed_clusters.pdf"), 
                     title='Computed classes', 
                     alpha=1.0, 
                     size=20)
        with open(os.path.join(outdir,"computed_clusters_2D.tsv"), "w") as filehandler: 
            for x,y,l in zip(reduced_data[:,0], 
                             reduced_data[:,1], 
                             labels):
                filehandler.write("{}\t{}\t{}\n".format(x,y,l))          
    
    # Plot the spots with colors corresponding to the predicted class
    # Use the HE image as background if the image is given
    n_col = min(num_columns, len(counts_table_files)) if joint_plot else 1
    n_row = max(int(len(counts_table_files) / n_col), 1) if joint_plot else 1
    if joint_plot:
        print("Generating a multiplot of {} rows and {} columns".format(n_row, n_col))
    for i, name in enumerate(counts_table_files):
        # Get the list of spot coordinates and colors to plot for each dataset
        x_points = spot_plot_data[i][0]
        y_points = spot_plot_data[i][1]
        colors_classes = spot_plot_data[i][2]
        colors_dimensionality = spot_plot_data[i][3]
        
        # Actually plot the data      
        outfile = os.path.join(outdir,
                               "{}_clusters.pdf".format(
                                          os.path.splitext(os.path.basename(name))[0])) if not joint_plot else None   
        scatter_plot(x_points=x_points, 
                     y_points=y_points,
                     colors=colors_classes,
                     output=outfile, 
                     alignment=None, 
                     cmap=None, 
                     title=name, 
                     xlabel=None, 
                     ylabel=None,
                     image=None, 
                     alpha=1.0, 
                     size=spot_size,
                     n_col=n_col,
                     n_row=n_row,
                     invert_y=True,
                     n_index=i+1 if joint_plot else 1)
        
    if joint_plot:
        fig = plt.gcf()
        fig.savefig(os.path.join(outdir,"joint_plot_clusters.pdf"), format='pdf', dpi=180)     
                                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--counts-files", required=True, nargs='+', type=str,
                        help="One or more matrices with gene counts per spot (genes as columns)")
    parser.add_argument("--normalization", default="RAW", metavar="[STR]", 
                        type=str, 
                        choices=["RAW", "REL", "CPM"],
                        help="Normalize the counts using:\n" \
                        "RAW = absolute counts\n" \
                        "REL = Each gene count divided by the total count of its spot\n" \
                        "CPM = Each gene count divided by the total count of its spot multiplied by its mean\n" \
                        "(default: %(default)s)")
    parser.add_argument("--num-clusters", default=None, metavar="[INT]", type=int, choices=range(2, 30),
                        help="The number of clusters/regions expected to be found.\n" \
                        "Note that this parameter has no effect with DBSCAN clustering.")
    parser.add_argument("--num-exp-genes", default=1, metavar="[FLOAT]", type=float,
                        help="The percentage of number of expressed genes (>= --min-gene-expression) a spot\n" \
                        "must have to be kept from the distribution of all expressed genes (default: %(default)s)")
    parser.add_argument("--num-exp-spots", default=1, metavar="[FLOAT]", type=float,
                        help="The percentage of number of expressed spots a gene\n" \
                        "must have to be kept from the total number of spots (default: %(default)s)")
    parser.add_argument("--min-gene-expression", default=1, type=int, metavar="[INT]", choices=range(1, 50),
                        help="The minimum count (number of reads) a gene must have in a spot to be\n"
                        "considered expressed (default: %(default)s)")
    parser.add_argument("--num-genes-keep", default=20, metavar="[INT]", type=int, choices=range(0, 99),
                        help="The percentage of genes to discard from the distribution of all the genes\n" \
                        "across all the spots using the variance or the top highest expressed\n" \
                        "(see --top-genes-criteria)\n " \
                        "Low variance or low expressed will be discarded (default: %(default)s)")
    parser.add_argument("--clustering", default="KMeans", metavar="[STR]", 
                        type=str, choices=["Hierarchical", "KMeans", "DBSCAN", "Gaussian"],
                        help="What clustering algorithm to use after the dimensionality reduction:\n" \
                        "Hierarchical = Hierarchical clustering (Ward)\n" \
                        "KMeans = Suitable for small number of clusters\n" \
                        "DBSCAN = Number of clusters will be automatically inferred\n" \
                        "Gaussian = Gaussian Mixtures Model\n" \
                        "(default: %(default)s)")
    parser.add_argument("--dimensionality", default="tSNE", metavar="[STR]", 
                        type=str, choices=["tSNE", "PCA", "ICA", "SPCA", "FactorAnalysis"],
                        help="What dimensionality reduction algorithm to use:\n" \
                        "tSNE = t-distributed stochastic neighbor embedding\n" \
                        "PCA = Principal component analysis\n" \
                        "ICA = Independent component analysis\n" \
                        "SPCA = Sparse principal component analysis\n" \
                        "FactorAnalysis = Linear model with Gaussian latent variables\n" \
                        "(default: %(default)s)")
    parser.add_argument("--use-log-scale", action="store_true", default=False,
                        help="Use log2(counts + 1) values in the dimensionality reduction step")
    parser.add_argument("--num-dimensions", default=2, metavar="[INT]", type=int, choices=range(2, 100),
                        help="The number of dimensions to use in the dimensionality reduction. (default: %(default)s)")
    parser.add_argument("--spot-size", default=20, metavar="[INT]", type=int, choices=range(1, 100),
                        help="The size of the spots when generating the plots. (default: %(default)s)")
    parser.add_argument("--top-genes-criteria", default="Variance", metavar="[STR]", 
                        type=str, choices=["Variance", "TopRanked"],
                        help="What criteria to use to keep top genes before doing\n" \
                        "the dimensionality reduction (Variance or TopRanked) (default: %(default)s)")
    parser.add_argument("--tsne-perplexity", default=30, metavar="[INT]", type=int, choices=range(5,500),
                        help="The value of the perplexity for the t-SNE method. (default: %(default)s)")
    parser.add_argument("--tsne-theta", default=0.5, metavar="[FLOAT]", type=float,
                        help="The value of theta for the t-SNE method. (default: %(default)s)")
    parser.add_argument("--tsne-initial-dims", default=50, metavar="[INT]", type=int,
                        help="The number of initial dimensions of the PCA step in the t-SNE clustering. (default: %(default)s)")
    parser.add_argument("--outdir", default=None, help="Path to output dir")
    parser.add_argument("--pca-auto-components", default=None, metavar="[FLOAT]", type=float,
                        help="For the PCA dimensionality reduction the number of dimensions\n" \
                        "are computed so to include the percentage of variance given as input [0.1-1].")
    parser.add_argument("--dbscan-min-size", default=5, metavar="[INT]", type=int, choices=range(5,500),
                        help="The value of the minimum cluster sizer for DBSCAN. (default: %(default)s)")
    parser.add_argument("--dbscan-eps", default=0.5, metavar="[FLOAT]", type=float,
                        help="The value of the EPS parameter for DBSCAN. (default: %(default)s)")
    parser.add_argument("--joint-plot", action="store_true", default=False, 
                        help="Generate one figure for all the datasets instead of one figure per dataset.")
    parser.add_argument("--num-columns", default=1, type=int, metavar="[INT]",
                        help="The number of columns when using --joint-plot (default: %(default)s)")
    args = parser.parse_args()
    main(args.counts_files, 
         args.normalization, 
         args.num_clusters,
         args.num_exp_genes,
         args.num_exp_spots,
         args.min_gene_expression,
         args.num_genes_keep,
         args.clustering, 
         args.dimensionality, 
         args.use_log_scale, 
         args.num_dimensions, 
         args.spot_size,
         args.top_genes_criteria,
         args.outdir,
         args.tsne_perplexity,
         args.tsne_theta,
         args.tsne_initial_dims,
         args.pca_auto_components,
         args.dbscan_min_size,
         args.dbscan_eps,
         args.joint_plot,
         args.num_columns)

