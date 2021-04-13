#! /usr/bin/env python
"""
This script performs a supervised training and prediction for
Spatial Transcriptomics datasets.

The multi-class classification can be performed with either SVC, NN or
logistic regression

The training set must be a matrix with counts (genes as columns and spots as rows)
and the test set must also be a matrix of counts with the same format.

One file with class/cluster labels for the training set is needed
so for the classifier to know what class/cluster each spot in
the training set belongs to, the file should be tab delimited :

SPOT_NAME(as it in the matrix) CLASS_NUMBER

The script will then try to predict the classes of the spots in the
test set. If class/cluster labels for the test sets
are given the script will compute accuracy of the prediction.

The script allows to normalize the train/test counts using different
methods as well as performing pre-filtering operations.

@Author Jose Fernandez Navarro <jc.fernandez.navarro@gmail.com>
"""
import argparse
import sys
import os
import pandas as pd
import pickle
from stanalysis.preprocessing import *
from stanalysis.utils import *
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


def main(train_data,
         test_data,
         train_classes_file,
         test_classes_file,
         log_scale,
         normalization,
         outdir,
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
         model_file,
         num_genes_keep_train, 
         num_genes_keep_test,
         top_genes_criteria_train, 
         top_genes_criteria_test):

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
        sys.stderr.write("Error, invalid number of expressed spots\n")
        sys.exit(1)

    if not outdir or not os.path.isdir(outdir):
        outdir = os.getcwd()
    print("Output folder {}".format(outdir))

    print("Loading training dataset...")
    train_data_frame = pd.read_csv(train_data, sep="\t", header=0,
                                   index_col=0, engine='c', low_memory=True)
    train_data_frame = remove_noise(train_data_frame, num_exp_genes, 
                                    num_exp_spots, min_gene_expression)
    # Load all the classes for the training set
    train_labels = parse_labels(train_classes_file, min_class_size)

    print("Loading prediction dataset...")
    test_data_frame = pd.read_csv(test_data, sep="\t", header=0,
                                  index_col=0, engine='c', low_memory=True)
    test_data_frame = remove_noise(test_data_frame, num_exp_genes, 
                                   num_exp_spots, min_gene_expression)
    # Load all the classes for the prediction set (if given)
    if test_classes_file is not None:
        test_labels = parse_labels(test_classes_file, 0)
    
    # Normalize counts
    print("Normalizing...")
    train_data_frame = normalize_data(train_data_frame, normalization)
    test_data_frame = normalize_data(test_data_frame, normalization)
    
    # Keep top genes (variance or expressed)
    train_data_frame = keep_top_genes(train_data_frame, num_genes_keep_train / 100.0, 
                                      criteria=top_genes_criteria_train)
    test_data_frame = keep_top_genes(test_data_frame, num_genes_keep_test / 100.0, 
                                     criteria=top_genes_criteria_test)
    
    # Keep only the record in the training set that intersects with the prediction set
    print("Genes in training set {}".format(train_data_frame.shape[1]))
    print("Spots in training set {}".format(train_data_frame.shape[0]))
    print("Genes in prediction set {}".format(test_data_frame.shape[1]))
    print("Spots in prediction set {}".format(test_data_frame.shape[0]))
    intersect_genes = np.intersect1d(train_data_frame.columns.values, 
                                     test_data_frame.columns.values)
    if len(intersect_genes) == 0:
        sys.stderr.write("Error, there are no genes intersecting the train and test datasets\n")
        sys.exit(1)

    print("Intersected genes {}".format(len(intersect_genes)))
    train_data_frame = train_data_frame.loc[:,intersect_genes]
    test_data_frame = test_data_frame.loc[:,intersect_genes]

    # Log the counts
    if log_scale:
        print("Transforming datasets to log space...")
        train_data_frame = np.log1p(train_data_frame)
        test_data_frame = np.log1p(test_data_frame)

    # Apply the z-transformation
    if standard_transformation:
        print("Applying standard transformation...")
        train_data_frame = ztransformation(train_data_frame)
        test_data_frame = ztransformation(test_data_frame)
        
    # Sort labels data together
    shared_spots = np.intersect1d(train_data_frame.index, train_labels.index)
    train_data_frame = train_data_frame.loc[shared_spots,:]
    train_labels = np.asarray(train_labels.loc[shared_spots, ["cluster"]]).ravel()
    if test_classes_file:
        shared_spots = np.intersect1d(test_data_frame.index, test_labels.index)
        test_data_frame = test_data_frame.loc[shared_spots,:]
        test_labels = np.asarray(test_labels.loc[shared_spots, ["cluster"]]).ravel()
    
    # Get the numpy counts
    train_counts = train_data_frame.astype(np.float32).values

    class_weight = "balanced" if stratified_sampler else None
    if model_file is None:
        # Train the classifier and predict
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
        elif classifier in "RF":
            print("Random Forest classifier")
            model = RandomForestClassifier(n_estimators=100, 
                                           criterion="gini",
                                           max_depth=None,
                                           min_samples_split=2,
                                           min_samples_leaf=1,
                                           min_weight_fraction_leaf=0.0,
                                           max_features="auto", 
                                           max_leaf_nodes=None, 
                                           min_impurity_decrease=0.0, 
                                           min_impurity_split=None, 
                                           bootstrap=True, 
                                           oob_score=False, 
                                           n_jobs=-1, 
                                           random_state=None, 
                                           warm_start=False, 
                                           class_weight=class_weight)
        elif classifier in "GB":
            print("Gradient Boosting classifier")
            model = GradientBoostingClassifier(loss="deviance",
                                               learning_rate=0.1, 
                                               n_estimators=100, 
                                               subsample=1.0, 
                                               criterion="friedman_mse",
                                               min_samples_split=2, 
                                               min_samples_leaf=1, 
                                               min_weight_fraction_leaf=0.0, 
                                               max_depth=3, 
                                               min_impurity_decrease=0.0, 
                                               min_impurity_split=None, 
                                               init=None, 
                                               random_state=None, 
                                               max_features=None, 
                                               max_leaf_nodes=None, 
                                               warm_start=False, 
                                               presort="auto", 
                                               validation_fraction=0.2, 
                                               n_iter_no_change=20, 
                                               tol=0.0001)
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
    try:
        pd.DataFrame(data=model.coef_,
                     index=sorted(set(train_labels.tolist())),
                     columns=intersect_genes).to_csv(os.path.join(outdir,
                                                                  "genes_contributions.tsv"),
                                                                  sep='\t')
    except:
        sys.stdout.write("Warning, model's weights could not be obtained\n")
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--train-data", required=True, type=str,
                        help="Path to the input training data file (matrix of counts, genes as columns)")
    parser.add_argument("--test-data", required=True, type=str,
                        help="Path to the test training data file (matrix of counts, genes as columns)")
    parser.add_argument("--train-classes", required=True, type=str,
                        help="Path to the training classes file (SPOT LABEL)")
    parser.add_argument("--test-classes", required=False, type=str,
                        help="Path to the test classes file (SPOT LABEL)")
    parser.add_argument("--model-file", required=False, type=str, default=None,
                        help="Path to saved model file to avoid recomputing the model and only predict")
    parser.add_argument("--log-scale", action="store_true", default=False,
                        help="Convert the training and test sets to log space (if no batch correction is performed)")
    parser.add_argument("--standard-transformation", action="store_true", default=False,
                        help="Apply the z-score transformation to each feature (gene)")
    parser.add_argument("--normalization", default="RAW", metavar="[STR]",
                        type=str,
                        choices=["RAW", "REL", "CPM"],
                        help="Normalize the counts using:\n"
                        "RAW = absolute counts\n"
                        "REL = Each gene count divided by the total count of its spot\n"
                        "CPM = Each gene count divided by the total count of its spot multiplied by its mean\n"
                        "(default: %(default)s)")
    parser.add_argument("--epochs", type=int, default=1000, metavar="[INT]",
                        help="The number of epochs to train (default: %(default)s)")
    parser.add_argument("--outdir", help="Path to output dir")
    parser.add_argument("--num-exp-genes", default=0.01, metavar="[FLOAT]", type=float,
                        help="The percentage of number of expressed genes (>= --min-gene-expression) a spot\n"
                        "must have to be kept from the distribution of all expressed genes (0.0 - 1.0) (default: %(default)s)")
    parser.add_argument("--num-exp-spots", default=0.01, metavar="[FLOAT]", type=float,
                        help="The percentage of number of expressed spots (>= --min-gene-expression) a gene\n"
                        "must have to be kept from the total number of spots (0.0 - 1.0) (default: %(default)s)")
    parser.add_argument("--min-gene-expression", default=1, type=float, metavar="[FLOAT]",
                        help="The minimum count a gene must have in a spot to be\n"
                        "considered expressed when filtering (default: %(default)s)")
    parser.add_argument("--classifier", default="SVC", metavar="[STR]",
                        type=str,
                        choices=["SVM", "LR", "NN", "GB", "RF"],
                        help="The classifier to use:\n"
                        "SVM = Support Vector Machine\n"
                        "LR = Logistic Regression\n"
                        "NN = Neural Network\n"
                        "GB = Gradient Boosting\n"
                        "RF = Random Forest\n"
                        "(default: %(default)s)")
    parser.add_argument("--svm-kernel", default="linear", metavar="[STR]",
                        type=str,
                        choices=["linear", "poly", "rbf", "sigmoid"],
                        help="What kernel to use with the SVM classifier:\n"
                        "linear = a linear kernel\n"
                        "poly = a polynomial kernel\n"
                        "rbf = a rbf kernel\n"
                        "sigmoid = a sigmoid kernel\n"
                        "(default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=200, metavar="[INT]",
                        help="The batch size for the Neural Network classifier (default: %(default)s)")
    parser.add_argument("--hidden-layers-size", type=int, nargs="+", metavar="[INT]", default=[1000, 500],
                        help="The sizes of the hidden layers for the Neural Network\n"
                        "The number of hidden layers will correspond to the number of sizes given (default: %(default)s)")
    parser.add_argument("--learning-rate", type=float, default=0.001, metavar="[FLOAT]",
                        help="The learning rate for the Neural Network classifier (default: %(default)s)")
    parser.add_argument("--stratified-sampler", action="store_true", default=False,
                        help="Draw samples with equal probabilities when training")
    parser.add_argument("--min-class-size", type=int, default=10, metavar="[INT]",
                        help="The minimum number of elements a class must has in the training set (default: %(default)s)")
    parser.add_argument("--num-genes-keep-train", default=50, metavar="[INT]", type=int, choices=range(0, 99),
                        help="The percentage of genes to discard from the distribution of all the genes\n"
                        "across all the spots using the variance or the top highest expressed\n"
                        "(see --top-genes-criteria-train)\n "
                        "Low variance or low expressed genes will be discarded (default: %(default)s)")
    parser.add_argument("--num-genes-keep-test", default=50, metavar="[INT]", type=int, choices=range(0, 99),
                        help="The percentage of genes to discard from the distribution of all the genes\n"
                        "across all the spots using the variance or the top highest expressed\n"
                        "(see --top-genes-criteria-test)\n "
                        "Low variance or low expressed genes will be discarded (default: %(default)s)")
    parser.add_argument("--top-genes-criteria-train", default="Variance", metavar="[STR]", 
                        type=str, choices=["Variance", "TopRanked"],
                        help="What criteria to use to reduce the number of genes (Variance or TopRanked) (default: %(default)s)")
    parser.add_argument("--top-genes-criteria-test", default="Variance", metavar="[STR]", 
                        type=str, choices=["Variance", "TopRanked"],
                        help="What criteria to use to reduce the number of genes (Variance or TopRanked) (default: %(default)s)")
    args = parser.parse_args()
    main(args.train_data,
         args.test_data,
         args.train_classes,
         args.test_classes,
         args.log_scale,
         args.normalization,
         args.outdir,
         args.standard_transformation,
         args.epochs,
         args.num_exp_genes,
         args.num_exp_spots,
         args.min_gene_expression,
         args.classifier,
         args.svm_kernel,
         args.batch_size,
         args.learning_rate,
         args.stratified_sampler,
         args.min_class_size,
         args.hidden_layers_size,
         args.model_file,
         args.num_genes_keep_train,
         args.num_genes_keep_test,
         args.top_genes_criteria_train,
         args.top_genes_criteria_test)

