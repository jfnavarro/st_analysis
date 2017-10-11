#! /usr/bin/env python
""" 
This script performs a supervised prediction in ST datasets
using a training set and a test set. 

The training set will be one or more matrices of
with counts (genes as columns and spots as rows)
and the test set will be one matrix of counts.

One file or files with class labels for the training set is needed
so the classifier knows what class each spot(row) in
the training set belongs to, the file should
be tab delimited :

SPOT_NAME(as it in the matrix) CLASS_NUMBER

It will then try to predict the classes of the spots(rows) in the 
test set. If class labels for the test sets
are given the script will compute accuracy of the prediction.

The script allows to normalize the train/test counts using different
methods.

The script will output the predicted classes and the spots
plotted on top of an image if the image is given.

@Author Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>
"""
import argparse
import sys
import os
import numpy as np
import pandas as pd
#from sklearn.feature_selection import VarianceThreshold
from stanalysis.preprocessing import *
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from stanalysis.visualization import scatter_plot, color_map
from stanalysis.alignment import parseAlignmentMatrix
from stanalysis.analysis import weighted_color, composite_colors
from cProfile import label
from matplotlib.colors import LinearSegmentedColormap

def main(train_data, 
         test_data, 
         classes_train, 
         classes_test,
         use_log_scale,
         normalization,
         outdir,
         alignment, 
         image,
         spot_size):

    if len(train_data) == 0 or any([not os.path.isfile(f) for f in train_data]) \
    or len(train_data) != len(classes_train) \
    or len(classes_train) == 0 or any([not os.path.isfile(f) for f in classes_train]) \
    or not os.path.isfile(classes_test):
        sys.stderr.write("Error, input file/s not present or invalid format\n")
        sys.exit(1)
     
    if not outdir or not os.path.isdir(outdir):
        outdir = os.getcwd()
        
    print("Output folder {}".format(outdir))
  
    # Merge input train datasets (Spots are rows and genes are columns)
    train_data_frame = aggregate_datatasets(train_data)
    train_genes = list(train_data_frame.columns.values)
    
    # loads all the classes for the training set
    train_labels_dict = dict()
    for i,labels_file in enumerate(classes_train):
        with open(labels_file) as filehandler:
            for line in filehandler.readlines():
                tokens = line.split()
                train_labels_dict["{}_{}".format(i,tokens[0])] = int(tokens[1])
    # make sure the spots in the training set data frame
    # and the label training spots have the same order
    # and are the same 
    train_labels = list()
    for spot in train_data_frame.index:
        try:
            train_labels.append(train_labels_dict[spot])
        except KeyError:
            train_data_frame.drop(spot, axis=0, inplace=True)
    if len(train_labels) != len(train_data_frame.index):
        sys.stderr.write("Error, none of the train labels were not found in the train data\n")
        sys.exit(1)
         
    # loads the test set
    # spots are rows and genes are columns
    test_data_frame = pd.read_table(test_data, sep="\t", header=0, index_col=0)    
    test_genes = list(test_data_frame.columns.values)
    
    # loads all the classes for the test set
    # filter out labels whose spot is not present and
    # also make sure that the order of the labels is the same in the data frame
    test_labels = list()
    if classes_test is not None:
        spot_label = dict()
        with open(classes_test) as filehandler:
            for line in filehandler.readlines():
                tokens = line.split()
                assert(len(tokens) == 2)
                spot_label[tokens[0]] = int(tokens[1])      
        for spot in test_data_frame.index:
            try:
                test_labels.append(spot_label[spot])
            except KeyError:
                test_data_frame.drop(spot, axis=0, inplace=True)
        if len(test_labels) != len(test_data_frame.index):
            sys.stderr.write("Error, none of the test labels were not found in the test data\n")
            sys.exit(1)  
          
    # Keep only the record in the training set that intersects with the test set
    print("Training genes {}".format(len(train_genes)))
    print("Test genes {}".format(len(test_genes)))
    intersect_genes = np.intersect1d(train_genes, test_genes)
    if len(intersect_genes) == 0:
        sys.stderr.write("Error, there are no genes intersecting the train and test datasets\n")
        sys.exit(1)  
            
    print("Intersected genes {}".format(len(intersect_genes)))
    train_data_frame = train_data_frame.ix[:,intersect_genes]
    test_data_frame = test_data_frame.ix[:,intersect_genes]
    
    # Classes in test and train must be the same
    print("Training elements {}".format(len(train_labels)))
    print("Test elements {}".format(len(test_labels)))
    print("Class labels {}".format(sorted(set(train_labels))))
    
    # Get the normalized counts
    train_data_frame = normalize_data(train_data_frame, normalization)
    test_data_frame = normalize_data(test_data_frame, normalization)
    test_counts = test_data_frame.values 
    train_counts = train_data_frame.values 
    
    # Log the counts
    if use_log_scale:
        train_counts = np.log2(train_counts + 1)
        test_counts = np.log2(test_counts + 1)
        
    # Train the classifier and predict
    # TODO optimize parameters of the classifier (kernel="rbf" or "sigmoid")
    classifier = OneVsRestClassifier(SVC(probability=True, random_state=0, 
                                         decision_function_shape="ovr", kernel="linear"), n_jobs=4)
    classifier = classifier.fit(train_counts, train_labels)
    predicted_class = classifier.predict(test_counts) 
    predicted_prob = classifier.predict_proba(test_counts)
     
    # Compute accuracy
    if classes_test is not None:
        print("Classification report for classifier {0}:\n{1}\n".
              format(classifier, metrics.classification_report(test_labels, predicted_class)))
        print("Confusion matrix:\n{}".format(metrics.confusion_matrix(test_labels, predicted_class)))
    
    # Write the spots and their predicted classes/probs to a file
    x_points = list()
    y_points = list()
    merged_prob_colors = list()
    unique_colors = [color_map[i] for i in set(sorted(predicted_class))]
    with open(os.path.join(outdir, "predicted_classes.txt"), "w") as filehandler:
        labels = list(test_data_frame.index)
        for i,label in enumerate(predicted_class):
            probs = predicted_prob[i].tolist()
            merged_prob_colors.append(composite_colors(unique_colors, probs))
            tokens = labels[i].split("x")
            assert(len(tokens) == 2)
            y = float(tokens[1])
            x = float(tokens[0])
            x_points.append(x)
            y_points.append(y)
            filehandler.write("{0}\t{1}\t{2}\n".format(labels[i], label,
                                                       "\t".join(['{:.6f}'.format(x) for x in probs])))
            
    # Plot the spots with the predicted color on top of the tissue image
    # The plotted color will be taken from a linear space from 
    # all the unique colors from the classes so it shows
    # how strong the prediction is for a specific spot
    # alignment_matrix will be identity if alignment file is None
    alignment_matrix = parseAlignmentMatrix(alignment)
    cm = LinearSegmentedColormap.from_list("CustomMap", unique_colors, N=100)
    scatter_plot(x_points=x_points, 
                 y_points=y_points, 
                 colors=merged_prob_colors, 
                 output=os.path.join(outdir,"predicted_classes_tissue_probability.pdf"), 
                 alignment=alignment_matrix, 
                 cmap=cm, 
                 title='Computed classes tissue (probability)', 
                 xlabel='X', 
                 ylabel='Y',
                 image=image, 
                 alpha=1.0, 
                 size=spot_size,
                 show_legend=False,
                 show_color_bar=False)
    # Plot also the predicted color for each spot (highest probablity)
    scatter_plot(x_points=x_points, 
                 y_points=y_points, 
                 colors=[int(c) for c in predicted_class], 
                 output=os.path.join(outdir,"predicted_classes_tissue.pdf"), 
                 alignment=alignment_matrix, 
                 cmap=None, 
                 title='Computed classes tissue', 
                 xlabel='X', 
                 ylabel='Y',
                 image=image, 
                 alpha=1.0, 
                 size=spot_size,
                 show_legend=True,
                 show_color_bar=False)
       
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--train-data", required=True, nargs='+', type=str,
                        help="One or more data frames with normalized counts")
    parser.add_argument("--test-data", required=True,
                        help="One data frame with normalized counts")
    parser.add_argument("--train-classes", required=True, nargs='+', type=str,
                        help="One of more files with the class of each spot in the train data as: XxY INT")
    parser.add_argument("--test-classes", default=None,
                        help="One file with the class of each spot in the test data as: XxY INT")
    parser.add_argument("--use-log-scale", action="store_true", default=False,
                        help="Use log2 + 1 for the training and test set instead of raw/normalized counts.")
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
                        "Scran = Deconvolution Sum Factors\n" \
                        "REL = Each gene count divided by the total count of its spot\n" \
                        "(default: %(default)s)")
    parser.add_argument("--alignment", default=None,
                        help="A file containing the alignment image " \
                        "(array coordinates to pixel coordinates) as a 3x3 matrix in tab delimited format\n" \
                        "This is only useful if you want to plot the image in original size or the image " \
                        "is not cropped to the array boundaries")
    parser.add_argument("--image", default=None, 
                        help="When given the data will plotted on top of the image, \
                        if the alignment matrix is given the data points will be transformed to pixel coordinates")
    parser.add_argument("--outdir", help="Path to output dir")
    parser.add_argument("--spot-size", default=20, metavar="[INT]", type=int, choices=range(1, 100),
                        help="The size of the spots when generating the plots. (default: %(default)s)")
    args = parser.parse_args()
    main(args.train_data, args.test_data, args.train_classes, 
         args.test_classes, args.use_log_scale, args.normalization, 
         args.outdir, args.alignment, args.image, args.spot_size)

