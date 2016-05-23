#! /usr/bin/env python
#@Author Jose Fernandez
""" 
This script performs a supervised prediction
using a training set and a test set. 
The training set will be a data frame
with normalized counts from single cell data
and the test set will also be a data frame with counts.
A file with class labels for the training set is needed
so the classifier knows what class each spot(row) in
the training set belongs to. It will then try
to predict the classes of the spots(rows) in the 
test set. If class labels for the test sets
are given the script will compute accuracy of the prediction.
The script will output the predicted classes and the spots
plotted on top of an image if the image is given.
"""
import argparse
import sys
import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from matplotlib import transforms
import tempfile
from stanalysis.visualization import plotSpotsWithImage

def main(train_data, 
         test_data, 
         classes_train, 
         classes_test, 
         outdir,
         alignment, 
         image):

    if not os.path.isfile(train_data) or not os.path.isfile(test_data) \
    or not os.path.isfile(classes_train) or not os.path.isfile(classes_test):
        sys.stderr.write("Error, input file/s not present or invalid format\n")
        sys.exit(1)
     
    if not outdir or not os.path.isdir(outdir):
        outdir = tempfile.mktemp()
        
    print "Out folder " + outdir
           
    # loads all the barcodes classes for the training set
    barcodes_classes_train = dict()
    with open(classes_train, "r") as filehandler:
        for line in filehandler.readlines():
            tokens = line.split()
            x = int(tokens[1])
            y = int(tokens[2])
            class_label = str(tokens[0])
            barcodes_classes_train[(x,y)] = class_label
       
    # loads all the barcodes classes for the test set
    barcodes_classes_test = dict()
    with open(classes_test, "r") as filehandler:
        for line in filehandler.readlines():
            tokens = line.split()
            x = int(tokens[1])
            y = int(tokens[2])
            class_label = str(tokens[0])
            barcodes_classes_test[(x,y)] = class_label[0]
      
    # loads the training set
    # spots are rows and genes are columns
    train_data_frame = pd.read_table(train_data, sep="\t", header=0, index_col=0)
    train_genes = list(train_data_frame.columns.values)
    # keep only the spots that are in both the train/test data and the labels/classes
    indexes_remove_train = [x for x in list(train_data_frame.index) if x not in barcodes_classes_train]
    train_data_frame = train_data_frame.drop(indexes_remove_train, axis=0)
    
    # loads the test set
    # spots are rows and genes are columns
    test_data_frame = pd.read_table(test_data, sep="\t", header=0, index_col=0)    
    test_genes = list(test_data_frame.columns.values)
    # keep only the spots that are in both the train/test data and the labels/classes
    indexes_remove_test = [x for x in list(test_data_frame.index) if x not in barcodes_classes_test]
    test_data_frame = test_data_frame.drop(indexes_remove_test, axis=0)
    
    # Keep only the record in the training set that intersects with the test set
    print "Training genes " + str(len(train_genes))
    print "Test genes " + str(len(test_genes))
    intersect_genes = np.intersect1d(train_genes, test_genes)
    print "Intersected genes " + str(len(intersect_genes))
    train_data_frame = train_data_frame.ix[:,intersect_genes]
    test_data_frame = test_data_frame.ix[:,intersect_genes]
    
    train_labels = [barcodes_classes_train[x] for x in list(train_data_frame.index)]
    test_labels = [barcodes_classes_test[x] for x in list(test_data_frame.index)]
    # Classes in test and train must be the same
    print "Training elements " + str(len(train_labels))
    print "Test elements " + str(len(test_labels))
    class_labels = sorted(set(train_labels))
    print "Class labels"
    print class_labels
    
    # Keep only 1000 highest scored genes (TODO)
    
    # Scale spots (columns) against the mean and variance (TODO)
    
    # Get the counts
    test_counts = test_data_frame.values # Assume they are normalized
    train_counts = train_data_frame.values # Assume they are normalized
    
    # Train the classifier and predict
    # TODO optimize parameters of the classifier
    classifier = OneVsRestClassifier(LinearSVC(random_state=0))
    predicted = classifier.fit(train_counts, train_labels).predict(test_counts) 
    
    # Compute accuracy
    print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(test_labels, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_labels, predicted)) 
    
    # Write the spots and their classes to a file
    x_points = list()
    y_points = list()
    with open(os.path.join(outdir, "predicted_classes.txt"), "w") as filehandler:
        labels = list(test_data_frame.index)
        for i,label in enumerate(predicted):
            bc = labels[i].split("x")
            assert(len(bc) == 2)
            x = bc[0]
            y = bc[1]
            x_points.append(int(x))
            y_points.append(int(y))
            filehandler.write(str(label) + "\t" + str(x) + "\t" + str(y) + "\n")
            
    # Plot the spots with the predicted color on top of the tissue image
    if image is not None and os.path.isfile(image):
        colors = [int(x) for x in predicted]
        plotSpotsWithImage(x_points, y_points, colors, image,
                            "computed_classes_tissue.png", alignment)
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", 
                        help="The data frame with the normalized counts for training")
    parser.add_argument("--test-data", 
                        help="The data frame with the normalized counts for testing")
    parser.add_argument("--train-classes", 
                        help="A tab delimited file mapping barcodes to their classes for training")
    parser.add_argument("--test-classes", default=None,
                        help="A tab delimited file mapping barcodes to their classes for testing")
    parser.add_argument("--alignment", 
                        help="Alignment matrix needed when using the image", 
                        nargs="+", type=float, default=None)
    parser.add_argument("--image", default=None, 
                        help="When given the data will plotted on top of the image, \
                        if the alignment matrix is given the data will be aligned")
    parser.add_argument("--outdir", help="Path to output dir")
    args = parser.parse_args()
    main(args.train_data, args.test_data, args.train_classes, 
         args.test_classes, args.outdir, 
         args.alignment, args.image)

