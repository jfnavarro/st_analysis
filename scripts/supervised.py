#! /usr/bin/env python
#@Author Jose Fernandez

import argparse
import sys
import os
import numpy as np
import pandas as pd
from stpipeline.common.utils import fileOk
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from matplotlib import transforms
import tempfile

def main(train_data, 
         test_data, 
         classes_train, 
         classes_test, 
         outdir,
         barcodes_file,
         alignment, 
         image):

    if not fileOk(train_data) or not fileOk(test_data) \
    or not fileOk(classes_train) or not fileOk(classes_test):
        sys.stderr.write("Error, input file/s not present or invalid format\n")
        sys.exit(-1)
     
    if not outdir or not os.path.isdir(outdir):
        outdir = tempfile.mktemp()
        
    print "Out folder " + outdir
           
    # loads all the barcodes classes for the training set
    barcodes_classes_train = dict()
    with open(classes_train, "r") as filehandler:
        for line in filehandler.readlines():
            tokens = line.split()
            barcodes_classes_train[tokens[1]] = tokens[0]
       
    # loads all the barcodes classes for the test set
    barcodes_classes_test = dict()
    with open(classes_test, "r") as filehandler:
        for line in filehandler.readlines():
            tokens = line.split()
            barcodes_classes_test[tokens[1]] = tokens[0]
      
    # loads the training set
    # Barcodes are rows and genes are columns
    train_data_frame = pd.read_table(train_data, sep="\t", header=0, index_col=0)
    train_genes = list(train_data_frame.columns.values)
    # keep only the spots that are in both the train/test data and the labels/classes
    indexes_remove_train = [x for x in list(train_data_frame.index) if x not in barcodes_classes_train]
    train_data_frame = train_data_frame.drop(indexes_remove_train, axis=0)
    
    # loads the test set
    # Barcodes are rows and genes are columns
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
    classifier = OneVsRestClassifier(LinearSVC(random_state=0))
    predicted = classifier.fit(train_counts, train_labels).predict(test_counts) 
    
    # Compute accuracy
    print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(test_labels, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_labels, predicted)) 
    
    # Write the spots and their classes to a file
    if barcodes_file is not None and os.path.isfile(barcodes_file):
        # First get the barcodes coordinates
        map_barcodes = dict()
        x_points = list()
        y_points = list()
        with open(barcodes_file, "r") as filehandler:
            for line in filehandler.readlines():
                tokens = line.split()
                map_barcodes[tokens[0]] = (tokens[1],tokens[2])
        # Write barcodes, labels and coordinates
        with open(os.path.join(outdir, "predicted_classes.txt"), "w") as filehandler:
            test_labels = list(test_data_frame.index)
            for i,label in enumerate(predicted):
                bc = test_labels[i]
                x,y = map_barcodes[bc]
                x_points.append(int(x))
                y_points.append(int(y))
                filehandler.write(str(label) + "\t" + bc + "\t" + str(x) + "\t" + str(y) + "\n")
    
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
                colors = [int(x) for x in predicted]
                img = plt.imread(image)
                fig = plt.figure(figsize=(8,8))
                a = fig.add_subplot(111, aspect='equal')
                base_trans = a.transData
                tr = transforms.Affine2D(matrix = alignment_matrix) + base_trans
                a.scatter(x_points, y_points, c=colors, edgecolor="none", s=50, transform=tr)   
                a.imshow(img)
                fig.set_size_inches(16, 16)
                fig.savefig(os.path.join(outdir,"computed_classes_tissue.png"), dpi=300)
                        
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
    main(args.train_data, args.test_data, args.train_classes, 
         args.test_classes, args.outdir, args.barcodes_file, 
         args.alignment, args.image)

