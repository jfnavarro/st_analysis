#! /usr/bin/env python
""" 
This script performs a supervised prediction
using a training set and a test set. 

The training set will be one or more matrices of
with counts (genes as columns and spots as rows)
and the test set will be one matrix of counts.

One file or files with class labels for the training set is needed
so the classifier knows what class each spot(row) in
the training set belongs to, the file should
be tab delimited :

CLASS_NUMBER SPOT_NAME(as it in the matrix)

It will then try to predict the classes of the spots(rows) in the 
test set. If class labels for the test sets
are given the script will compute accuracy of the prediction.

The script assumes that the counts are already normalized.

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
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from stanalysis.visualization import scatter_plot
from stanalysis.alignment import parseAlignmentMatrix
from cProfile import label

def get_classes_coordinate(class_file):
    """ Helper function
    to get a dictionary of spot -> class 
    from a tab delimited file
    """
    barcodes_classes = dict()
    with open(class_file, "r") as filehandler:
        for line in filehandler.readlines():
            tokens = line.split()
            assert(len(tokens) == 2)
            spot = tokens[1]
            class_label = tokens[0]
            barcodes_classes[spot] = class_label
    return barcodes_classes
               
def main(train_data, 
         test_data, 
         classes_train, 
         classes_test,
         use_log_scale,
         outdir,
         alignment, 
         image):

    if len(train_data) == 0 or any([not os.path.isfile(f) for f in train_data]) \
    or len(train_data) != len(classes_train) \
    or len(classes_train) == 0 or any([not os.path.isfile(f) for f in classes_train]) \
    or not os.path.isfile(classes_test):
        sys.stderr.write("Error, input file/s not present or invalid format\n")
        sys.exit(1)
     
    if not outdir or not os.path.isdir(outdir):
        outdir = os.getcwd()
        
    print "Output folder {}".format(outdir)
  
    # Merge input train datasets (Spots are rows and genes are columns)
    train_data_frame = aggregate_datatasets(train_data)
    train_genes = list(train_data_frame.columns.values)
    
    # loads all the classes for the training set
    train_labels = list()
    for labels_file in classes_train:
        with open(labels_file) as filehandler:
            for line in filehandler.readlines():
                train_labels.append(line.split()[0])
    assert(len(train_labels) == len(train_data_frame))       
         
    # loads the test set
    # spots are rows and genes are columns
    test_data_frame = pd.read_table(test_data, sep="\t", header=0, index_col=0)    
    test_genes = list(test_data_frame.columns.values)
    
    # loads all the classes for the test set
    # filter out labels whose barcode is not present and
    # also make sure that the order of the labels is the same in the data frame
    test_labels = list()
    if classes_test is not None:
        spot_label = dict()
        with open(classes_test) as filehandler:
            for line in filehandler.readlines():
                tokens = line.split()
                spot = tokens[1]
                label = tokens[0]
                spot_label[spot] = label      
        for spot in test_data_frame.index:
            try:
                test_labels.append(spot_label[spot])
            except KeyError:
                test_data_frame.drop(spot, axis=0, inplace=True)
        assert(len(test_labels) == len(test_data_frame))  
      
          
    # Keep only the record in the training set that intersects with the test set
    print "Training genes {}".format(len(train_genes))
    print "Test genes {}".format(len(test_genes))
    intersect_genes = np.intersect1d(train_genes, test_genes)
    print "Intersected genes {}".format(len(intersect_genes))
    train_data_frame = train_data_frame.ix[:,intersect_genes]
    test_data_frame = test_data_frame.ix[:,intersect_genes]
    
    # Classes in test and train must be the same
    print "Training elements {}".format(len(train_labels))
    print "Test elements {}".format(len(test_labels))
    class_labels = sorted(set(train_labels))
    print "Class labels"
    print class_labels
    
    # Keep only 1000 highest scored genes (TODO)
    # Scale spots (columns) against the mean and variance (TODO)
    
    # Get the counts
    test_counts = test_data_frame.values # Assume they are normalized
    train_counts = train_data_frame.values # Assume they are normalized
    
    # Log the counts
    if use_log_scale:
        train_counts = np.log2(train_counts + 1)
        test_counts = np.log2(test_counts + 1)
        
    # Train the classifier and predict
    # TODO optimize parameters of the classifier
    classifier = OneVsRestClassifier(LinearSVC(random_state=0), n_jobs=4)
    # NOTE one could also get the predict prob of each class for each spot predict_proba() 
    predicted = classifier.fit(train_counts, train_labels).predict(test_counts) 
    
    # Compute accuracy
    if classes_test is not None:
        print("Classification report for classifier {0}:\n{1}\n".
              format(classifier, metrics.classification_report(test_labels, predicted)))
        print("Confusion matrix:\n{}".format(metrics.confusion_matrix(test_labels, predicted)))
    
    # Write the spots and their predicted classes to a file
    x_points = list()
    y_points = list()
    with open(os.path.join(outdir, "predicted_classes.txt"), "w") as filehandler:
        labels = list(test_data_frame.index)
        for i,label in enumerate(predicted):
            tokens = labels[i].split("x")
            assert(len(tokens) == 2)
            y = float(tokens[1])
            x = float(tokens[0])
            x_points.append(x)
            y_points.append(y)
            filehandler.write("{0}\t{1}\t{2}\n".format(label, x, y))
            
    # Plot the spots with the predicted color on top of the tissue image
    colors = [int(x) for x in predicted]
    # alignment_matrix will be identity if alignment file is None
    alignment_matrix = parseAlignmentMatrix(alignment)
    scatter_plot(x_points=x_points, 
                 y_points=y_points, 
                 colors=colors, 
                 output=os.path.join(outdir,"predicted_classes_tissue.png"), 
                 alignment=alignment_matrix, 
                 cmap=None, 
                 title='Computed classes tissue', 
                 xlabel='X', 
                 ylabel='Y',
                 image=image, 
                 alpha=1.0, 
                 size=200)
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-data", required=True, nargs='+', type=str,
                        help="One or more data frames with normalized counts")
    parser.add_argument("--test-data", required=True,
                        help="One data frame with normalized counts")
    parser.add_argument("--train-classes", required=True, nargs='+', type=str,
                        help="One of more files with the class of each spot in the train data")
    parser.add_argument("--test-classes", default=None,
                        help="One file with the class of each spot in the train data")
    parser.add_argument("--use-log-scale", action="store_true", default=False,
                        help="Use log values for the counts.")
    parser.add_argument("--alignment", default=None,
                        help="A file containing the alignment image " \
                        "(array coordinates to pixel coordinates) as a 3x3 matrix in tab delimited format\n" \
                        "This is only useful if you want to plot the image in original size or the image " \
                        "is not cropped to the array boundaries")
    parser.add_argument("--image", default=None, 
                        help="When given the data will plotted on top of the image, \
                        if the alignment matrix is given the data points will be transformed to pixel coordinates")
    parser.add_argument("--outdir", help="Path to output dir")
    args = parser.parse_args()
    main(args.train_data, args.test_data, args.train_classes, 
         args.test_classes, args.use_log_scale, args.outdir, 
         args.alignment, args.image)

