#! /usr/bin/env python
""" 
This script performs a supervised training and prediction for ST datasets

The multi-class classification can be performed with either SVC, NN or 
logistic regression

The training set will be a matrix with counts (genes as columns and spots as rows)
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
import gc

from stanalysis.preprocessing import *
from stanalysis.utils import *

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier

from cProfile import label

def main(train_data,
         test_data,
         train_classes_file, 
         test_classes_file, 
         log_scale, 
         normalization, 
         outdir, 
         batch_correction, 
         standard_transformation, 
         epochs, 
         num_exp_genes, 
         num_exp_spots, 
         min_gene_expression, 
         classifier, 
         svm_kernel,
         batch_size, 
         learning_rate, 
         stratified_sampler, 
         min_class_size,
         hidden_layers_size,
         model_file):

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
        
    if model_file is not None and not os.path.isfile(model_file):
        sys.stderr.write("Error, invalid model file\n")
        sys.exit(1)
       
    if normalization == "Scran" and log_scale:
        sys.stderr.write("Warning, when performing Scran normalization log-scale will be ignored\n")
     
    if batch_correction and log_scale:
        sys.stderr.write("Warning, when performing batch correction log-scale will be ignored\n")
         
    if epochs < 1:
        sys.stderr.write("Error, number of epoch is too small\n")
        sys.exit(1)
                 
    if min_class_size < 0:
        sys.stderr.write("Error, invalid minimum class size\n")
        sys.exit(1)

    if learning_rate < 0:
        sys.stderr.write("Error, learning rate\n")
        sys.exit(1)
        
    if batch_size < 1:
        sys.stderr.write("Error, batch size is too small\n")
        sys.exit(1)
    
    if num_exp_genes < 0.0 or num_exp_genes > 1.0:
        sys.stderr.write("Error, invalid number of expressed genes\n")
        sys.exit(1)
        
    if num_exp_spots < 0.0 or num_exp_spots > 1.0:
        sys.stderr.write("Error, invalid number of expressed genes\n")
        sys.exit(1)
                  
    if not outdir or not os.path.isdir(outdir):
        outdir = os.getcwd()   
    print("Output folder {}".format(outdir))
    
    print("Loading training dataset...")
    train_data_frame = pd.read_table(train_data, sep="\t", header=0, index_col=0,
                                     engine='c', low_memory=True)
    train_data_frame = remove_noise(train_data_frame, 1.0, num_exp_spots, min_gene_expression)
    train_genes = list(train_data_frame.columns.values)
    
    # Load all the classes for the training set
    train_labels_dict = load_labels(train_classes_file)
    train_data_frame, train_labels = update_labels(train_data_frame, train_labels_dict)
    
    print("Loading prediction dataset...")
    test_data_frame = pd.read_table(test_data, sep="\t", header=0, index_col=0,
                                    engine='c', low_memory=True)
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
    train_data_frame = train_data_frame.loc[:,intersect_genes]
    test_data_frame = test_data_frame.loc[:,intersect_genes]
    
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
        batch_corrected = computeMnnBatchCorrection([b.transpose() for b in 
                                                     [train_data_frame,test_data_frame]])
        train_data_frame = batch_corrected[0].transpose()
        test_data_frame = batch_corrected[1].transpose()
        train_data_frame.to_csv(os.path.join(outdir, "train_bc_final.tsv"), sep="\t")
        test_data_frame.to_csv(os.path.join(outdir, "test_bc_final.tsv"), sep="\t")
        del batch_corrected
        gc.collect()
        
    # Log the counts
    if log_scale and not batch_correction and not normalization == "Scran":
        print("Transforming datasets to log space...")
        train_data_frame = np.log1p(train_data_frame)
        test_data_frame = np.log1p(test_data_frame)
        
    # Apply the z-transformation
    if standard_transformation:
        print("Applying standard transformation...")
        train_data_frame = ztransformation(train_data_frame)
        test_data_frame = ztransformation(test_data_frame)

    # Update labels again
    train_data_frame, train_labels = update_labels(train_data_frame, train_labels_dict)
    if test_classes_file is not None:
        test_data_frame, test_labels = update_labels(test_data_frame, test_labels_dict)
    
    # Discard "noisy" classes
    print("Removing classes with less than {} elements".format(min_class_size))
    train_data_frame, train_labels = filter_classes(train_data_frame, train_labels, min_class_size)
    print("Training set {}".format(train_data_frame.shape[0]))
    
    # Get the numpy counts
    train_counts = train_data_frame.astype(np.float32).values
    del train_data_frame
    gc.collect()
        
    if model_file is None:
        # Train the classifier and predict
        class_weight = "balanced" if stratified_sampler else None
        if classifier in "SVC":
            print("One vs rest SVM")
            model = OneVsRestClassifier(SVC(probability=True, 
                                            random_state=None,
                                            tol=0.001,
                                            max_iter=epochs,
                                            class_weight=class_weight,
                                            decision_function_shape="ovr", 
                                            kernel=svm_kernel), n_jobs=-1)
        elif classifier in "NN":
            print("Neural Network with the following hidden layers {}".format(
                " ".join([str(x) for x in hidden_layers_size])))
            model = MLPClassifier(hidden_layer_sizes=hidden_layers_size, 
                                  activation='tanh', 
                                  solver='adam', 
                                  alpha=0.0001, 
                                  batch_size=batch_size, 
                                  learning_rate_init=learning_rate, 
                                  max_iter=epochs, 
                                  shuffle=True, 
                                  random_state=None, 
                                  tol=0.0001, 
                                  momentum=0.9)
        else:
            print("One vs rest Logistic Regression")
            model = OneVsRestClassifier(LogisticRegression(penalty='l2', 
                                                           dual=False, 
                                                           tol=0.0001, 
                                                           C=1.0, 
                                                           fit_intercept=True, 
                                                           intercept_scaling=1, 
                                                           class_weight=class_weight, 
                                                           random_state=None, 
                                                           solver='lbfgs', 
                                                           max_iter=epochs, 
                                                           multi_class='multinomial', 
                                                           warm_start=False), n_jobs=-1)
        # Training and validation
        print("Training the model...")
        model = model.fit(train_counts, train_labels)
        # Save the model
        pickle.dump(model, open(os.path.join(outdir, "model.sav"), 'wb'))
        print("Model trained and saved!")
    else:
        print("Loading model {}".format(model_file))
        model = pickle.load(open(model_file, 'rb'))
        
    # Predict
    print("Predicting on test data..")
    predict_counts = test_data_frame.astype(np.float32).values
    test_index = test_data_frame.index
    del test_data_frame
    gc.collect()
    
    predicted_class = model.predict(predict_counts)  
    predicted_prob = model.predict_proba(predict_counts)
    
    # Compute accuracy
    if test_classes_file is not None:
        print("Classification report\n{}".
              format(metrics.classification_report(test_labels, predicted_class)))
        print("Confusion matrix:\n{}".format(metrics.confusion_matrix(test_labels, predicted_class)))
    
    with open(os.path.join(outdir, "predicted_classes.tsv"), "w") as filehandler:
        for spot, pred, probs in zip(test_index, predicted_class, predicted_prob):
            filehandler.write("{0}\t{1}\t{2}\n".format(spot, pred,
                                                       "\t".join(['{:.4f}'.format(x) for x in probs.tolist()]))) 
            
    # Print the weights for each gene
    pd.DataFrame(data=model.coef_,
                 index=sorted(set([index_label_map[x] for x in train_labels])),
                 columns=intersect_genes).to_csv(os.path.join(outdir,
                                                              "genes_contributions.tsv"), 
                                                              sep='\t')

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
                        help="Path to the test classes file (SPOT LABEL)")
    parser.add_argument("--model-file", required=False, type=str, default=None,
                        help="Path to saved model file to avoid recomputing the model and only predict")
    parser.add_argument("--log-scale", action="store_true", default=False,
                        help="Convert the training and test sets to log space (if no batch correction is performed)")
    parser.add_argument("--batch-correction", action="store_true", default=False,
                        help="Perform batch-correction (Scran::Mnncorrect()) between train and test sets")
    parser.add_argument("--standard-transformation", action="store_true", default=False,
                        help="Apply the z-score transformation to each feature (gene)")
    parser.add_argument("--normalization", default="RAW", metavar="[STR]", 
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
                        choices=["SVM", "LR", "NN"],
                        help="The classifier to use:\n" \
                        "SVM = Support Vector Machine\n" \
                        "LR = Logistic Regression\n" \
                        "NN = Neural Network\n" \
                        "(default: %(default)s)")
    parser.add_argument("--svm-kernel", default="linear", metavar="[STR]", 
                        type=str, 
                        choices=["linear", "poly", "rbf", "sigmoid"],
                        help="What kernel to use with the SVM classifier:\n" \
                        "linear = a linear kernel\n" \
                        "poly = a polynomial kernel\n" \
                        "rbf = a rbf kernel\n" \
                        "sigmoid = a sigmoid kernel\n" \
                        "(default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=1000, metavar="[INT]",
                        help="The batch size for the Neural Network classifier (default: %(default)s)")
    parser.add_argument("--hidden-layers-size", type=int, nargs="+", metavar="[INT]", default=[1000, 500],
                        help="The sizes of the hidden layers for the Neural Network\n " \
                        "The number of hidden layers will correspond to the number of sizes given (default: %(default)s)")
    parser.add_argument("--learning-rate", type=float, default=0.001, metavar="[FLOAT]",
                        help="The learning rate for the Neural Network classifier (default: %(default)s)")
    parser.add_argument("--stratified-sampler", action="store_true", default=False,
                        help="Draw samples with equal probabilities when training")
    parser.add_argument("--min-class-size", type=int, default=10, metavar="[INT]",
                        help="The minimum number of elements a class must has in the training set (default: %(default)s)")
    args = parser.parse_args()
    main(args.train_data, args.test_data, args.train_classes, 
         args.test_classes, args.log_scale, args.normalization, 
         args.outdir, args.batch_correction, args.standard_transformation, 
         args.epochs, args.num_exp_genes, args.num_exp_spots, 
         args.min_gene_expression, args.classifier, args.svm_kernel,
         args.batch_size, args.learning_rate, args.stratified_sampler, 
         args.min_class_size, args.hidden_layers_size, args.model_file)

