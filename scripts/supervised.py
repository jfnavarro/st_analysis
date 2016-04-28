#! /usr/bin/env python
#@Author Jose Fernandez

import argparse
import sys
import os
import numpy as np
import pandas as pd
from stpipeline.common.utils import fileOk
from sklearn.feature_selection import VarianceThreshold
sys.path.append('./')
from cyclone import cyclone
import tempfile

def main(train_data, test_data, classes_train, classes_test, signatures, outdir):

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
    train_data_frame = pd.read_table(train_data, sep="\t", header=0, index_col=0)
    if signatures is not None and os.path.isfile(signatures):
        with open(signatures,"r") as filehander:
            signatures = [sign for sign in filehandler.readlines()]
            train_data_frame = train_data_frame[signatures]
    #else:
    #    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    #    train_data_frame = sel.fit_transform(train_data_frame)
    train_genes = list(train_data_frame.columns.values)
    train_labels = list(train_data_frame.index)
    train_labels_updated = list()
    indexes_remove_train = list()
    for i,label in enumerate(train_labels):
        try:
            train_labels_updated.append(barcodes_classes_train[label])
        except KeyError,e:
            print "Barcode not found in the training class file " + str(e)
            indexes_remove_train.append(i)
    train_data_frame = train_data_frame.drop(train_data_frame.index[indexes_remove_train])
    train_counts = train_data_frame.values # Assume they are normalized
    
    # loads the test set
    test_data_frame = pd.read_table(test_data, sep="\t", header=0, index_col=0)    
    test_genes = list(test_data_frame.columns.values)
    test_labels = list(test_data_frame.index)
    test_labels_updated = list()
    indexes_remove_test = list()
    for i,label in enumerate(test_labels):
        try:
            test_labels_updated.append(barcodes_classes_test[label])
        except KeyError,e:
            print "Barcode not found in the test class file " + str(e)
            indexes_remove_test.append(i)
    test_data_frame = test_data_frame.drop(test_data_frame.index[indexes_remove_test])
    test_counts = test_data_frame.values # Assume they are normalized
    
    # Classes in test and train must be the same
    print "Training elements " + str(len(train_labels_updated))
    print "Test elements " + str(len(test_labels_updated))
    assert(set(train_labels_updated) == set(test_labels_updated))
    class_labels = sorted(set(train_labels_updated))
    print "Class labels"
    print class_labels
    # Convert train and test labels to the index of the class labels
    # TODO really refactor this
    new_train_labels = []
    for label in train_labels_updated:
        new_train_labels.append(class_labels.index(label) + 1)
    new_test_labels = []
    for label in test_labels_updated:
        new_test_labels.append(class_labels.index(label) + 1)
    
    # creates intersection of genes
    print "Training genes " + str(len(train_genes))
    print "Test genes " + str(len(test_genes))
    intersect_genes = np.intersect1d(train_genes, test_genes)
    print "Intersected genes " + str(len(intersect_genes))

    # create model
    # Data frame is expected to have genes as columns (TODO double check this)
    model = cyclone(train_counts, 
                    row_namesY=np.array(train_genes),
                    cc_geneNames=np.array(intersect_genes),
                    labels=np.array(new_train_labels), 
                    Y_tst=test_counts, 
                    row_namesY_tst=np.array(test_genes), 
                    labels_tst=np.array(new_test_labels),
                    norm="none")
    # train model
    # CV can be 2-10 or LOOCV
    model.trainModel(rftop=40, cv=10, out_dir=outdir, do_pca=True, npc=2)
    
    # plot results
    model.plotScatter(class_labels=class_labels, out_dir=outdir, xlab="", ylab="", method='RF')
    model.plotScatter(class_labels=class_labels, out_dir=outdir, xlab="", ylab="", method='LR')
    model.plotScatter(class_labels=class_labels, out_dir=outdir, xlab="", ylab="", method='LRall')
    model.plotScatter(class_labels=class_labels, out_dir=outdir, xlab="", ylab="", method='GNB')
    model.plotF1(out_dir=outdir, class_labels=class_labels)
    model.plotPerformance(out_dir=outdir, method='RF', perfType="ROC", class_labels=class_labels)
    model.plotPerformance(out_dir=outdir, method='LR', perfType="ROC", class_labels=class_labels)
    model.plotPerformance(out_dir=outdir, method='LRall', perfType="ROC", class_labels=class_labels)
    model.plotPerformance(out_dir=outdir, method='GNB', perfType="ROC", class_labels=class_labels)
    model.writePrediction(out_dir=outdir, method='GNB', original_test_labels=test_labels, output="GNB_prediction.txt")
    model.writePrediction(out_dir=outdir, method='LR', original_test_labels=test_labels, output="LR_prediction.txt")
    model.writePrediction(out_dir=outdir, method='RF', original_test_labels=test_labels, output="RF_prediction.txt")
    model.writePrediction(out_dir=outdir, method='LRall', original_test_labels=test_labels, output="LRall_prediction.txt")
                
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
    parser.add_argument("--signatures", default=None,
                        help="a list of genes to be used as signatures")
    parser.add_argument("--outdir", help="Path to output dir")
    args = parser.parse_args()
    main(args.train_data, args.test_data, args.train_classes, 
         args.test_classes, args.signatures, args.outdir)

