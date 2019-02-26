#! /usr/bin/env python
""" 
This script performs a supervised training and prediction for ST datasets

The multi-label classification can be performed with either SVC or 
logistic regression

The training set will be a matrix
with counts (genes as columns and spots as rows)
and the test set will be a matrix of counts with the same format

One file with class labels for the training set is needed
so the classifier knows what class each spot(row) in
the training set belongs to, the file should
be tab delimited :

SPOT_NAME(as it in the matrix) CLASS_NUMBER

It will then try to predict the classes of the spots(rows) in the 
test set. If class labels for the test sets
are given the script will compute accuracy of the prediction.

The script allows to normalize the train/test counts using different
methods as well as pre-filtering operations.

@Author Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>
"""
import argparse
import sys
import os
import numpy as np
import pandas as pd
import pickle

from stanalysis.preprocessing import *
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from cProfile import label

def update_labels(counts, labels_dict):
    # make sure the spots in the training set data frame
    # and the label training spots have the same order
    # and are the same 
    train_labels = list()
    for spot in counts.index:
        try:
            train_labels.append(labels_dict[spot])
        except KeyError:
            counts.drop(spot, axis=0, inplace=True)
    assert(len(train_labels) == counts.shape[0])
    return counts, train_labels   
    
def load_labels(filename):
    labels_dict = dict()
    with open(filename) as filehandler:
        for line in filehandler.readlines():
            tokens = line.split()
            assert(len(tokens) == 2)
            labels_dict[tokens[0]] = int(tokens[1])
    return labels_dict

def main(train_data,
         test_data,
         train_classes_file, 
         test_classes_file, 
         use_log_scale, 
         normalization, 
         outdir, 
         batch_correction, 
         z_transformation, 
         epochs, 
         num_exp_genes, 
         num_exp_spots, 
         min_gene_expression, 
         classifier, 
         svc_kernel):

    if not os.path.isfile(train_data):
        sys.stderr.write("Error, the training data input is not valid\n")
        sys.exit(1)
    
    if not os.path.isfile(train_classes_file):
        sys.stderr.write("Error, the train labels input is not valid\n")
        sys.exit(1)
     
    if not os.path.isfile(test_data):
        sys.stderr.write("Error, the test data input is not valid\n")
        sys.exit(1)
    
    if test_classes_file is not None and not os.path.isfile(test_classes_file):
        sys.stderr.write("Error, the test labels input is not valid\n")
        sys.exit(1)
       
    if normalization == "Scran" and use_log_scale:
        sys.stderr.write("Warning, Scran normalization converts to log space already\n")
                 
    if not outdir or not os.path.isdir(outdir):
        outdir = os.getcwd()   
    print("Output folder {}".format(outdir))
    
    print("Loading training dataset...")
    train_data_frame = pd.read_table(train_data, sep="\t", header=0, index_col=0)
    train_data_frame = remove_noise(train_data_frame, 1.0, num_exp_spots, min_gene_expression)
    train_genes = list(train_data_frame.columns.values)
    
    # Load all the classes for the training set
    train_labels_dict = load_labels(train_classes_file)
    train_data_frame, train_labels = update_labels(train_data_frame, train_labels_dict)
    
    print("Loading prediction dataset...")
    test_data_frame = pd.read_table(test_data, sep="\t", header=0, index_col=0)
    test_data_frame = remove_noise(test_data_frame, 1.0, num_exp_spots, min_gene_expression)
    test_genes = list(test_data_frame.columns.values)
    
    # Load all the classes for the prediction set
    if test_classes_file is not None:
        test_labels_dict = load_labels(test_classes_file)
        test_data_frame, test_labels = update_labels(test_data_frame, test_labels_dict)
          
    # Keep only the record in the training set that intersects with the prediction set
    print("Genes in training set {}".format(train_data_frame.shape[1]))
    print("Spots in training set {}".format(train_data_frame.shape[0]))
    print("Genes in prediction set {}".format(test_data_frame.shape[1]))
    print("Spots in prediction set {}".format(test_data_frame.shape[0]))
    intersect_genes = np.intersect1d(train_genes, test_genes)
    if len(intersect_genes) == 0:
        sys.stderr.write("Error, there are no genes intersecting the train and test datasets\n")
        sys.exit(1)  
            
    print("Intersected genes {}".format(len(intersect_genes)))
    train_data_frame = train_data_frame.ix[:,intersect_genes]
    test_data_frame = test_data_frame.ix[:,intersect_genes]
    
    # Get the normalized counts (prior removing noisy spots/genes)
    train_data_frame = remove_noise(train_data_frame, num_exp_genes, 1.0, min_gene_expression)
    train_data_frame = normalize_data(train_data_frame, normalization,
                                      adjusted_log=normalization == "Scran")
    
    test_data_frame = remove_noise(test_data_frame, num_exp_genes, 1.0, min_gene_expression)
    test_data_frame = normalize_data(test_data_frame, normalization, 
                                     adjusted_log=normalization == "Scran")
    
    # Perform batch correction (Batches are training and prediction set)
    if batch_correction:
        print("Performing batch correction...")
        batches = [b.transpose() for b in [train_data_frame,test_data_frame]]
        batch_corrected = computeMnnBatchCorrection(batches)
        train_data_frame = batch_corrected[0].transpose()
        test_data_frame = batch_corrected[1].transpose()
        
    # Apply the z-transformation
    if z_transformation:
        print("Applying z-score transformation...")
        train_data_frame = ztransformation(train_data_frame)
        test_data_frame = ztransformation(test_data_frame)

    # Update labels again
    train_data_frame, train_labels = update_labels(train_data_frame, train_labels_dict)
    if test_classes_file is not None:
        test_data_frame, test_labels = update_labels(test_data_frame, test_labels_dict)
       
    # Get the numpy counts
    train_counts = train_data_frame.astype(np.float32).values
    test_counts = test_data_frame.astype(np.float32).values
  
    # Log the counts
    if use_log_scale and not batch_correction and not normalization == "Scran":
        print("Transforming datasets to log space (log2 + 1)...")
        train_counts = np.log2(train_counts + 1)
        test_counts = np.log2(test_counts + 1)
        
    # Train the classifier and predict
    # TODO optimize parameters of the classifier (kernel="rbf" or "sigmoid")
    if classifier in "SVC":
        classifier = OneVsRestClassifier(SVC(probability=True, 
                                             random_state=0,
                                             tol=0.001,
                                             max_iter=epochs,
                                             decision_function_shape="ovr", 
                                             kernel=svc_kernel), n_jobs=-1)
    else:
        classifier = OneVsRestClassifier(LogisticRegression(penalty='l2', 
                                                            dual=False, 
                                                            tol=0.0001, 
                                                            C=1.0, 
                                                            fit_intercept=True, 
                                                            intercept_scaling=1, 
                                                            class_weight=None, 
                                                            random_state=0, 
                                                            solver='liblinear', 
                                                            max_iter=epochs, 
                                                            multi_class='ovr', 
                                                            warm_start=False), n_jobs=-1)
    classifier = classifier.fit(train_counts, train_labels)
    predicted_class = classifier.predict(test_counts) 
    predicted_prob = classifier.predict_proba(test_counts)
    
    # Save the model
    pickle.dump(classifier, open(os.path.join(outdir, "model.sav"), 'wb'))
    
    # Print the weights for each gene
    pd.DataFrame(data=classifier.coef_,
                 index=sorted(set(train_labels)),
                 columns=intersect_genes).to_csv(os.path.join(outdir,
                                                              "genes_contributions.tsv"), 
                                                              sep='\t')
    
    # Compute accuracy
    if test_classes_file is not None:
        print("Classification report\n{}".
              format(metrics.classification_report(test_labels, predicted_class)))
        print("Confusion matrix:\n{}".format(metrics.confusion_matrix(test_labels, predicted_class)))
    
    with open(os.path.join(outdir, "predicted_classes.tsv"), "w") as filehandler:
        for spot, pred, probs in zip(test_data_frame.index, predicted_class, predicted_prob):
            filehandler.write("{0}\t{1}\t{2}\n".format(spot, np.asscalar(pred),
                                                       "\t".join(['{:.6f}'.format(x) for x in probs.tolist()]))) 
       
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--train-data", required=True, type=str,
                        help="Path to the input training data file (matrix of counts, spots as rows)")
    parser.add_argument("--test-data", required=True, type=str,
                        help="Path to the test training data file (matrix of counts, spots as rows)")
    parser.add_argument("--train-classes", required=True, type=str,
                        help="Path to the training classes file (SPOT LABEL)")
    parser.add_argument("--test-classes", required=False, type=str,
                        help="Path to the test classes file (SPOT LABEL) (OPTIONAL)")
    parser.add_argument("--use-log-scale", action="store_true", default=False,
                        help="Use log2 + 1 for the training and test (if no batch correction is performed)")
    parser.add_argument("--batch-correction", action="store_true", default=False,
                        help="Perform batch-correction (Scran::Mnncorrect()) between train and test sets")
    parser.add_argument("--z-transformation", action="store_true", default=False,
                        help="Apply the z-score transformation to each spot (Sij - Mean(i) / std(i))")
    parser.add_argument("--normalization", default="DESeq2", metavar="[STR]", 
                        type=str, 
                        choices=["RAW", "DESeq2",  "REL", "Scran"],
                        help="Normalize the counts using:\n" \
                        "RAW = absolute counts\n" \
                        "DESeq2 = DESeq2::estimateSizeFactors(counts)\n" \
                        "Scran = Deconvolution Sum Factors (Marioni et al)\n" \
                        "REL = Each gene count divided by the total count of its spot\n" \
                        "(default: %(default)s)")
    parser.add_argument("--epochs", type=int, default=1000, metavar="[INT]",
                        help="The number of epochs to train (default: %(default)s)")
    parser.add_argument("--outdir", help="Path to output dir")
    parser.add_argument("--num-exp-genes", default=0.01, metavar="[FLOAT]", type=float,
                        help="The percentage of number of expressed genes (>= --min-gene-expression) a spot\n" \
                        "must have to be kept from the distribution of all expressed genes (0.0 - 1.0) (default: %(default)s)")
    parser.add_argument("--num-exp-spots", default=0.01, metavar="[FLOAT]", type=float,
                        help="The percentage of number of expressed spots a gene\n" \
                        "must have to be kept from the total number of spots (0.0 - 1.0) (default: %(default)s)")
    parser.add_argument("--min-gene-expression", default=1, type=int, metavar="[INT]", choices=range(1, 50),
                        help="The minimum count (number of reads) a gene must have in a spot to be\n"
                        "considered expressed (default: %(default)s)")
    parser.add_argument("--classifier", default="SVC", metavar="[STR]", 
                        type=str, 
                        choices=["SVC", "LR"],
                        help="The classifier to use:\n" \
                        "SVC = Support Vector Machine\n" \
                        "LR = Logistic Regression\n" \
                        "(default: %(default)s)")
    parser.add_argument("--svc-kernel", default="linear", metavar="[STR]", 
                        type=str, 
                        choices=["linear", "poly", "rbf", "sigmoid"],
                        help="What kernel to use with the SVC classifier:\n" \
                        "linear = a linear kernel\n" \
                        "poly = a polynomial kernel\n" \
                        "rbf = a rbf kernel\n" \
                        "sigmoid = a sigmoid kernel\n" \
                        "(default: %(default)s)")
    args = parser.parse_args()
    main(args.train_data, args.test_data, args.train_classes, 
         args.test_classes, args.use_log_scale, args.normalization, 
         args.outdir, args.batch_correction, args.z_transformation, 
         args.epochs, args.num_exp_genes, args.num_exp_spots, 
         args.min_gene_expression, args.classifier, args.svc_kernel)

