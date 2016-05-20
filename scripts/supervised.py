#! /usr/bin/env python
#@Author Jose Fernandez

import argparse
import sys
import os
import numpy as np
import pandas as pd
from stpipeline.common.utils import fileOk
from sklearn.feature_selection import VarianceThreshold
from sklearn import svm, metrics
import tempfile

def main(train_data, test_data, classes_train, classes_test, outdir):

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
    train_labels = list(train_data_frame.index)
    # keep only the genes that are in the given classes
    train_labels_updated = list()
    indexes_remove_train = list()
    for i,label in enumerate(train_labels):
        try:
            train_labels_updated.append(barcodes_classes_train[label])
        except KeyError,e:
            print "Barcode not found in the training class file " + str(e)
            indexes_remove_train.append(i)
    train_data_frame = train_data_frame.drop(train_data_frame.index[indexes_remove_train], axis=0, inplace=True)
    
    
    # loads the test set
    # Barcodes are rows and genes are columns
    test_data_frame = pd.read_table(test_data, sep="\t", header=0, index_col=0)    
    test_genes = list(test_data_frame.columns.values)
    test_labels = list(test_data_frame.index)
    # keep only the genes that are in the given classes
    test_labels_updated = list()
    indexes_remove_test = list()
    for i,label in enumerate(test_labels):
        try:
            test_labels_updated.append(barcodes_classes_test[label])
        except KeyError,e:
            print "Barcode not found in the test class file " + str(e)
            indexes_remove_test.append(i)
    test_data_frame = test_data_frame.drop(test_data_frame.index[indexes_remove_test], axis=0, inplace=True)
    
    # Classes in test and train must be the same
    print "Training elements " + str(len(train_labels_updated))
    print "Test elements " + str(len(test_labels_updated))
    assert(set(train_labels_updated) == set(test_labels_updated))
    class_labels = sorted(set(train_labels_updated))
    print "Class labels"
    print class_labels
    
    # Convert train and test labels to the integer index of the class labels
    new_train_labels = []
    for label in train_labels_updated:
        new_train_labels.append(class_labels.index(label) + 1)
    new_test_labels = []
    for label in test_labels_updated:
        new_test_labels.append(class_labels.index(label) + 1)
    
    # Keep only the record in the training set that intersects with the test set
    print "Training genes " + str(len(train_genes))
    print "Test genes " + str(len(test_genes))
    intersect_genes = np.intersect1d(train_genes, test_genes)
    print "Intersected genes " + str(len(intersect_genes))
    test_data_frame = test_data_frame.drop(test_data_frame.columns[intersect_genes], axis=1, inplace=True)
    train_data_frame = test_data_frame.drop(train_data_frame.columns[intersect_genes], axis=1, inplace=True)
    
    # Keep only 1000 highest scored genes
    
    # Get the counts
    test_counts = test_data_frame.values # Assume they are normalized
    train_counts = train_data_frame.values # Assume they are normalized
    
    # Train the classifier
    clf = svm.SVC(gamma=0.001, C=100.)
    clf.fit(train_counts, new_train_labels)  
    # Predict the test set
    predicted = clf.predict(test_counts)
    # Compute accuracy
    print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(new_test_labels, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(new_test_labels, predicted))

    #counts_and_predictions = list(zip(test_counts, predicted))
    #for index, (image, prediction) in enumerate(counts_and_predictions[:4]):
    #    plt.subplot(2, 4, index + 5)
    #    plt.axis('off')
    #    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    #    plt.title('Prediction: %i' % prediction)
    #plt.show()
    
                
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
    parser.add_argument("--outdir", help="Path to output dir")
    args = parser.parse_args()
    main(args.train_data, args.test_data, args.train_classes, 
         args.test_classes, args.signatures, args.outdir)

