#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
A script that does unsupervised learning on 
Spatial Transcriptomics datasets (matrix of counts)
It takes a list of datasets as input and outputs (for each given input):

 - a scatter plot with the predicted classes (coulored) for each spot 
 - the spots plotted onto the images (if given) with the predicted class/color
 - a file containing two columns (SPOT and CLASS) for each dataset

The input data frames must have the gene names as columns and
the spots coordinates as rows (1x1).

The user can select what clustering algorithm to use
and what dimensionality reduction technique to use and normalization
method to use. 

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
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from stanalysis.visualization import scatter_plot, scatter_plot3d, histogram
from stanalysis.preprocessing import *
from stanalysis.alignment import parseAlignmentMatrix
from stanalysis.analysis import Rtsne, linear_conv, computeNClusters
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
         alignment_files, 
         image_files, 
         num_dimensions, 
         spot_size,
         top_genes_criteria,
         outdir,
         use_adjusted_log,
         tsne_perplexity,
         tsne_theta,
         color_space_plots):

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
         
    if use_adjusted_log and use_log_scale:
        sys.stdout.write("Warning, both log and adjusted log are enabled " \
                         "only adjusted log will be used\n")
        use_log_scale = False
      
    if tsne_theta < 0.0 or tsne_theta > 1.0:
        sys.stdout.write("Warning, invalid value for theta. Using default..\n")
        tsne_theta = 0.5
                 
    if num_exp_genes <= 0 or num_exp_spots <= 0:
        sys.stdout.write("Error, min_exp_genes and min_exp_spots must be > 0.\n")
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
    counts = remove_noise(counts, num_exp_genes / 100.0, num_exp_spots / 100.0, 
                          min_expression=min_gene_expression)

    if len(counts.index) < 5 or len(counts.columns) < 10:
        sys.stdout.write("Error, too many spots/genes were filtered.\n")
        sys.exit(1) 
                
    # Normalize data
    print("Computing per spot normalization...")
    center_size_factors = not use_adjusted_log
    norm_counts = normalize_data(counts, normalization, 
                                 center=center_size_factors, adjusted_log=use_adjusted_log)

    # Keep top genes (variance or expressed)
    norm_counts = keep_top_genes(norm_counts, num_genes_keep / 100.0, criteria=top_genes_criteria)
       
    # Compute the expected number of clusters
    if num_clusters is None:
        num_clusters = computeNClusters(counts)
        print("Computation of number of clusters obtained {} clusters".format(num_clusters))
        
    if use_log_scale:
        print("Using pseudo-log counts log2(counts + 1)")
        norm_counts = np.log2(norm_counts + 1)  
      
    print("Performing dimensionality reduction...") 
           
    if "tSNE" in dimensionality:
        # NOTE the Scipy tsne seems buggy so we use the R one instead
        reduced_data = Rtsne(norm_counts, num_dimensions, theta=tsne_theta, perplexity=tsne_perplexity)
    elif "PCA" in dimensionality:
        # n_components = None, number of mle to estimate optimal
        decomp_model = PCA(n_components=num_dimensions, whiten=True, copy=True)
    elif "ICA" in dimensionality:
        decomp_model = FastICA(n_components=num_dimensions, 
                               algorithm='parallel', whiten=True,
                               fun='logcosh', w_init=None, random_state=None)
    elif "SPCA" in dimensionality:
        decomp_model = SparsePCA(n_components=num_dimensions, alpha=1)
    else:
        sys.stderr.write("Error, incorrect dimensionality reduction method\n")
        sys.exit(1)
     
    if not "tSNE" in dimensionality:
        # Perform dimensionality reduction, outputs a bunch of 2D/3D coordinates
        reduced_data = decomp_model.fit_transform(norm_counts)
    
    print("Performing clustering...")
    # Do clustering of the dimensionality reduced coordinates
    if "KMeans" in clustering:
        labels = KMeans(init='k-means++',
                        n_clusters=num_clusters,
                        n_init=10).fit_predict(reduced_data)
    elif "Hierarchical" in clustering:
        labels = AgglomerativeClustering(n_clusters=num_clusters,
                                         affinity='euclidean',
                                         linkage='ward').fit_predict(reduced_data)
    elif "DBSCAN" in clustering:
        labels = DBSCAN(eps=0.5, min_samples=5, 
                        metric='euclidean', n_jobs=-1).fit_predict(reduced_data)
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
        
    # We do not want zeroes in the labels
    if 0 in labels: labels = labels + 1

    # Compute a color_label based on the RGB representation of the 
    # 2D/3D dimensionality reduced coordinates
    labels_colors = list()
    x_max = max(reduced_data[:,0])
    x_min = min(reduced_data[:,0])
    y_max = max(reduced_data[:,1])
    y_min = min(reduced_data[:,1])
    x_p = reduced_data[:,0]
    y_p = reduced_data[:,1]
    z_p = y_p
    if num_dimensions == 3:
        z_p = reduced_data[:,2]
        z_max = max(reduced_data[:,2])
        z_min = min(reduced_data[:,2])
    for x,y,z in zip(x_p,y_p,z_p):
        r = linear_conv(x, x_min, x_max, 0.0, 1.0)
        g = linear_conv(y, y_min, y_max, 0.0, 1.0)
        b = linear_conv(z, z_min, z_max, 0.0, 1.0) if num_dimensions == 3 else 1.0
        labels_colors.append((r,g,b))

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
        else:
            sys.stderr.write("Error, the spots in the input data have "
                             "the wrong format {}\n.".format(spot))
            sys.exit(1)
        index = int(tokens2[0])
        spot_plot_data[index][0].append(x)
        spot_plot_data[index][1].append(y)
        spot_plot_data[index][2].append(labels[i])
        spot_plot_data[index][3].append(labels_colors[i])
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
    for i, name in enumerate(counts_table_files):
        # Get the list of spot coordinates and colors to plot for each dataset
        x_points = spot_plot_data[i][0]
        y_points = spot_plot_data[i][1]
        colors_classes = spot_plot_data[i][2]
        colors_dimensionality = spot_plot_data[i][3]

        # Retrieve alignment matrix and image if any
        image = image_files[i] if image_files is not None \
        and len(image_files) >= i else None
        alignment = alignment_files[i] if alignment_files is not None \
        and len(alignment_files) >= i else None
        
        # alignment_matrix will be identity if alignment file is None
        alignment_matrix = parseAlignmentMatrix(alignment)
        
        # Actually plot the data         
        scatter_plot(x_points=x_points, 
                     y_points=y_points,
                     colors=colors_classes,
                     output=os.path.join(outdir,
                                         "{}_clusters.pdf".format(
                                          os.path.splitext(os.path.basename(name))[0])), 
                     alignment=alignment_matrix, 
                     cmap=None, 
                     title=name, 
                     xlabel='X', 
                     ylabel='Y',
                     image=image, 
                     alpha=1.0, 
                     size=spot_size)
        if color_space_plots:
            scatter_plot(x_points=x_points, 
                         y_points=y_points,
                         colors=colors_dimensionality, 
                         output=os.path.join(outdir,
                                             "{}_color_space.pdf".format(
                                             os.path.splitext(os.path.basename(name))[0])), 
                         alignment=alignment_matrix, 
                         cmap=plt.get_cmap("hsv"), 
                         title=name, 
                         xlabel='X', 
                         ylabel='Y',
                         image=image, 
                         alpha=1.0, 
                         size=spot_size)        
                                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--counts-table-files", required=True, nargs='+', type=str,
                        help="One or more matrices with gene counts per feature/spot (genes as columns)")
    parser.add_argument("--normalization", default="DESeq2", metavar="[STR]", 
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
                        "REL = Each gene count divided by the total count of its spot\n" \
                        "(default: %(default)s)")
    parser.add_argument("--num-clusters", default=None, metavar="[INT]", type=int, choices=range(2, 16),
                        help="The number of clusters/regions expected to be found.\n" \
                        "If not given the number of clusters will be computed.\n" \
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
                        "Hierarchical = Hierarchical Clustering (Ward)\n" \
                        "KMeans = Suitable for small number of clusters\n" \
                        "DBSCAN = Number of clusters will be automatically inferred\n" \
                        "Gaussian = Gaussian Mixtures Model\n" \
                        "(default: %(default)s)")
    parser.add_argument("--dimensionality", default="tSNE", metavar="[STR]", 
                        type=str, choices=["tSNE", "PCA", "ICA", "SPCA"],
                        help="What dimensionality reduction algorithm to use:\n" \
                        "tSNE = t-distributed stochastic neighbor embedding\n" \
                        "PCA = Principal Component Analysis\n" \
                        "ICA = Independent Component Analysis\n" \
                        "SPCA = Sparse Principal Component Analysis\n" \
                        "(default: %(default)s)")
    parser.add_argument("--use-log-scale", action="store_true", default=False,
                        help="Use log2(counts + 1) values in the dimensionality reduction step")
    parser.add_argument("--alignment-files", default=None, nargs='+', type=str,
                        help="One or more tab delimited files containing and alignment matrix for the images as\n" \
                        "\t a11 a12 a13 a21 a22 a23 a31 a32 a33\n" \
                        "Only useful is the image has extra borders, for instance not cropped to the array corners\n" \
                        "or if you want the keep the original image size in the plots.")
    parser.add_argument("--image-files", default=None, nargs='+', type=str,
                        help="When provided the data will plotted on top of the image\n" \
                        "It can be one ore more, ideally one for each input dataset\n " \
                        "It is desirable that the image is cropped to the array\n" \
                        "corners otherwise an alignment file is needed")
    parser.add_argument("--num-dimensions", default=2, metavar="[INT]", type=int, choices=[2,3],
                        help="The number of dimensions to use in the dimensionality " \
                        "reduction (2 or 3). (default: %(default)s)")
    parser.add_argument("--spot-size", default=20, metavar="[INT]", type=int, choices=range(1, 100),
                        help="The size of the spots when generating the plots. (default: %(default)s)")
    parser.add_argument("--top-genes-criteria", default="Variance", metavar="[STR]", 
                        type=str, choices=["Variance", "TopRanked"],
                        help="What criteria to use to keep top genes before doing\n" \
                        "the dimensionality reduction (Variance or TopRanked) (default: %(default)s)")
    parser.add_argument("--use-adjusted-log", action="store_true", default=False,
                        help="Use adjusted log normalized counts (R Scater::normalized())\n"
                        "in the dimensionality reduction step (recommended with SCRAN normalization)")
    parser.add_argument("--tsne-perplexity", default=30, metavar="[INT]", type=int, choices=range(5,500),
                        help="The value of the perplexity for the t-sne method. (default: %(default)s)")
    parser.add_argument("--tsne-theta", default=0.5, metavar="[FLOAT]", type=float,
                        help="The value of theta for the t-sne method. (default: %(default)s)")
    parser.add_argument("--outdir", default=None, help="Path to output dir")
    parser.add_argument("--color-space-plots", action="store_true", default=False,
                        help="Generate also plots using the representation in color space of the\n" \
                        "dimensionality reduced coordinates")   
    args = parser.parse_args()
    main(args.counts_table_files, 
         args.normalization, 
         args.num_clusters,
         args.num_exp_genes,
         args.num_exp_spots,
         args.min_gene_expression,
         args.num_genes_keep,
         args.clustering, 
         args.dimensionality, 
         args.use_log_scale, 
         args.alignment_files, 
         args.image_files, 
         args.num_dimensions, 
         args.spot_size,
         args.top_genes_criteria,
         args.outdir,
         args.use_adjusted_log,
         args.tsne_perplexity,
         args.tsne_theta,
         args.color_space_plots)

