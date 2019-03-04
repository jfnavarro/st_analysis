#! /usr/bin/env python
# -*- coding: utf-8 -*-
""" 
This script performs a supervised training and prediction for ST datasets

The multi-label classification is performed using a 2 layers
neural network with the option to use CUDA

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

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

from stanalysis.preprocessing import *

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as utils
import torchvision

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit

import gc

# Windows work-around
__spec__ = None
import multiprocessing

SEED = 666

def computeWeightsClasses(dataset):
    # Distribution of labels
    label_count = defaultdict(int)
    for _,label in dataset:
        label_count[label.item()] += 1         
    # Weight for each sample
    weights = [1.0 / x for x in label_count.values()]
    return torch.DoubleTensor(weights)
    
def split_dataset(dataset, labels, split_num=0.8, min_size=50):
    train_indexes = list()
    test_indexes = list()
    train_labels = list()
    test_labels = list()
    label_indexes = dict()
    # Store indexes for each cluster
    for i,label in enumerate(labels):
        try:
            label_indexes[label].append(i)
        except KeyError:
            label_indexes[label] = [i]
    # Split randomly indexes for each cluster into training and testing sets  
    for label,indexes in label_indexes.items():
        if len(indexes) >= min_size:
            indexes = np.asarray(indexes)
            np.random.shuffle(indexes)
            cut = int(split_num * len(indexes))
            training, test = indexes[:cut], indexes[cut:]
            train_indexes += training.tolist()
            test_indexes += test.tolist()
            train_labels += [label] * len(training)
            test_labels += [label] * len(test)
    # Return the splitted datasets and their labels
    return dataset.iloc[train_indexes,:], dataset.iloc[test_indexes,:], train_labels, test_labels

def test(model, tst_loader, loss, use_cuda, verbose=False):
    model.eval()
    test_loss = 0
    preds = list()
    for data, target in tst_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            #data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += loss(output, target).item()
            _, pred = torch.max(output, 1)
            preds += pred.cpu().numpy().tolist()
    if verbose:
        print("Test set avg. loss: {:.4f}".format(test_loss / len(tst_loader.dataset)))
    return preds

def predict(model, X_pre, use_cuda):
    model.eval()
    data = X_pre
    if use_cuda:
        data = data.cuda()
    with torch.no_grad():
        data = Variable(data)
        output = model(data)
        _, pred = torch.max(output, 1)
    return output, pred

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
         train_batch_size,
         test_batch_size, 
         epochs, 
         learning_rate,
         stratified_sampler,
         min_class_size,
         use_cuda,
         num_exp_genes, 
         num_exp_spots,
         min_gene_expression,
         verbose):

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
         
    if not torch.cuda.is_available() and use_cuda:
        sys.stderr.write("Error, CUDA is not available in this computer\n")
        sys.exit(1)
          
    if not outdir or not os.path.isdir(outdir):
        outdir = os.getcwd()   
    print("Output folder {}".format(outdir))
    
    print("Loading training dataset...")
    train_data_frame = pd.read_table(train_data, sep="\t", header=0, index_col=0).astype(np.float32)
    # Remove noisy genes
    train_data_frame = remove_noise(train_data_frame, 1.0, num_exp_spots, min_gene_expression)
    train_genes = list(train_data_frame.columns.values)
    
    # Load all the classes for the training set
    train_labels_dict = load_labels(train_classes_file)
    train_data_frame, train_labels = update_labels(train_data_frame, train_labels_dict)
    
    print("Loading prediction dataset...")
    test_data_frame = pd.read_table(test_data, sep="\t", header=0, index_col=0).astype(np.float32)
    # Remove noisy genes
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
    
    # Get the normalized counts (prior removing noisy spot)
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
        train_data_frame.to_csv(os.path.join(outdir, "train_data_bc.tsv"), sep="\t")
        test_data_frame.to_csv(os.path.join(outdir, "test_data_bc.tsv"), sep="\t")
        
    # Apply the z-transformation
    if z_transformation:
        print("Applying z-score transformation...")
        train_data_frame = ztransformation(train_data_frame)
        test_data_frame = ztransformation(test_data_frame)

    # Update labels again
    train_data_frame, train_labels = update_labels(train_data_frame, train_labels_dict)
    if test_classes_file is not None:
        test_data_frame, test_labels = update_labels(test_data_frame, test_labels_dict)
            
    # Update labels so to ensure they go for 0-N sequentially
    labels_index_map = dict()
    index_label_map = dict()
    for i,label in enumerate(sorted(set(train_labels))):
        labels_index_map[label] = i
        index_label_map[i] = label
    print("Mapping of labels:")
    print(index_label_map)
    train_labels = [labels_index_map[x] for x in train_labels]
    
    # Split train and test dasasets
    print("Splitting training set into training and test sets (equally balancing clusters)")
    train_counts_x, train_counts_y, train_labels_x, train_labels_y = split_dataset(train_data_frame, 
                                                                                   train_labels, 0.7, min_class_size)
    
    print("Training set {}".format(train_counts_x.shape[0]))
    print("Test set {}".format(train_counts_y.shape[0]))
    
    # PyTorch needs floats
    train_counts = train_counts_x.astype(np.float32).values
    test_counts = train_counts_y.astype(np.float32).values
    predict_counts = test_data_frame.astype(np.float32).values
    
    # Log the counts
    if use_log_scale and not batch_correction and not normalization == "Scran":
        print("Transforming datasets to log space (log2 + 1)...")
        train_counts = np.log2(train_counts + 1)
        test_counts = np.log2(test_counts + 1)
        predict_counts = np.log2(predict_counts + 1)
        
    # Input and output sizes
    n_feature = train_counts.shape[1]
    n_ele_train = train_counts.shape[0]
    n_ele_test = test_counts.shape[0]
    n_class = max(set(train_labels)) + 1
    
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # workers = multiprocessing.cpu_count() - 1
    # In Windows we can only use few workers
    workers = 4
    print("Workers {}".format(workers))
    kwargs = {'num_workers': workers, 'pin_memory': True}
    
    # Set the SEED
    torch.manual_seed(SEED)
    if use_cuda:
        torch.cuda.manual_seed(SEED)

    # Create Tensor Flow train dataset
    X_train = torch.tensor(train_counts)
    X_test = torch.tensor(test_counts)
    
    y_train = torch.from_numpy(np.asarray(train_labels_x, dtype=np.longlong))
    y_test = torch.from_numpy(np.asarray(train_labels_y, dtype=np.longlong))
    
    # Create tensor datasets (train + test)
    trn_set = utils.TensorDataset(X_train, y_train)
    tst_set = utils.TensorDataset(X_test, y_test)
    
    # Create loaders with balanced labels
    if train_batch_size >= (n_ele_train / 2):
        print("The training batch size is almost as big as the training set...")
    if test_batch_size >= (n_ele_test / 2):
        print("The test batch size is almost as big as the test set...")
    if stratified_sampler:
        print("Using a stratified sampler for training set...")
        weights_train = computeWeightsClasses(trn_set)
        trn_sampler = utils.sampler.WeightedRandomSampler(weights_train, len(weights_train)) 
    else:
        trn_sampler = None    
    trn_loader = utils.DataLoader(trn_set, sampler=trn_sampler, 
                                  batch_size=train_batch_size, **kwargs)
    tst_loader = utils.DataLoader(tst_set, sampler=None, 
                                  batch_size=test_batch_size, **kwargs)

    # Init model
    H1 = 2000
    H2 = 500
    print("Creating NN model...")
    print("Input size {}".format(n_feature))
    print("Hidden layer 1 size {}".format(H1))
    print("Hidden layer 2 size {}".format(H2))
    print("Output size {}".format(n_class))
    model = torch.nn.Sequential(
        torch.nn.Linear(n_feature, H1),
        torch.nn.BatchNorm1d(num_features=H1),
        torch.nn.ReLU(),
        torch.nn.Linear(H1, H2),
        torch.nn.BatchNorm1d(num_features=H2),
        torch.nn.ReLU(),
        torch.nn.Linear(H2, n_class),
    )
    model.to(device) 
    
    # Creating loss
    loss = torch.nn.CrossEntropyLoss().cuda() if use_cuda else torch.nn.CrossEntropyLoss()
    
    # Creating optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    best_epoch_idx = -1
    best_f1 = 0.0
    history = list()
    best_model = dict()
    print("Training the model...")
    for epoch in range(epochs):
        if verbose:
            print('Epoch: {}'.format(epoch))
        # Train the model
        model.train()
        training_loss = 0
        training_f1 = 0
        for data, target in trn_loader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            output = model(data)
            tloss = loss(output, target)
            training_loss += tloss.item()
            # Backward pass
            tloss.backward()
             # Compute prediction's score
            with torch.no_grad():
                _, pred = torch.max(output, 1)
                training_f1 += f1_score(target.data.cpu().numpy(),
                                        pred.data.cpu().numpy(), 
                                        average='micro')
            # Update parameters
            optimizer.step()
        if verbose:
            print("Training set avg. loss: {:.4f}".format(training_loss / len(trn_loader.dataset)))
            print("Training set avg. micro-f1: {:.4f}".format(training_f1 / len(trn_loader.dataset)))
        # Testing
        preds = test(model, tst_loader, loss, use_cuda, verbose)
        conf_mat = confusion_matrix(train_labels_y, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(train_labels_y, preds, average='weighted')
        if verbose:
            print("Testing confusion matrix:\n", conf_mat)
            print("Teting precison {:.4f}\nRecall {:.4f}\nf1 {:.4f}\n".format(precision,recall,f1))  
        history.append((conf_mat, precision, recall, f1))
        if f1 > best_f1: 
            best_f1 = f1
            best_epoch_idx = epoch
            best_model = model.state_dict()

    print("Model trained!")
    print("Best epoch {}".format(best_epoch_idx))
    conf_mat, precision, recall, f1 = history[best_epoch_idx]
    print("Confusion matrix:\n", conf_mat)
    print("Precison {:.4f}\nRecall {:.4f}\nf1 {:.4f}\n".format(precision,recall,f1))    

    # Predict
    print("Predicting on test data..")
    model.load_state_dict(best_model)
    torch.save(model, os.path.join(outdir, "model.pt"))
    X_pre = torch.tensor(predict_counts)
    y_pre = test_labels if test_classes_file is not None else None
    out, preds = predict(model, X_pre, use_cuda)
    # Map labels back to their original value
    preds = [index_label_map[np.asscalar(x)] for x in preds.cpu().numpy()]
    if y_pre is not None:
        print("Classification report\n{}".
              format(classification_report(y_pre, preds)))
        print("Confusion matrix:\n{}".format(confusion_matrix(y_pre, preds)))
    with open(os.path.join(outdir, "predicted_classes.tsv"), "w") as filehandler:
        for spot, pred, probs in zip(test_data_frame.index, preds, out.cpu().numpy()):
            filehandler.write("{0}\t{1}\t{2}\n".format(spot, pred,
                                                       "\t".join(['{:.6f}'.format(x) for x in probs]))) 
        
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
    parser.add_argument("--normalization", default="RAW", metavar="[STR]", 
                        type=str, 
                        choices=["RAW", "DESeq2",  "REL", "Scran"],
                        help="Normalize the counts using:\n" \
                        "RAW = absolute counts\n" \
                        "DESeq2 = DESeq2::estimateSizeFactors(counts)\n" \
                        "Scran = Deconvolution Sum Factors (Marioni et al)\n" \
                        "REL = Each gene count divided by the total count of its spot\n" \
                        "(default: %(default)s)")
    parser.add_argument("--train-batch-size", type=int, default=1000, metavar="[INT]",
                        help="The input batch size for training (default: %(default)s)")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="[INT]",
                        help="The input batch size for testing (default: %(default)s)")
    parser.add_argument("--epochs", type=int, default=50, metavar="[INT]",
                        help="The number of epochs to train (default: %(default)s)")
    parser.add_argument("--learning-rate", type=float, default=0.001, metavar="[FLOAT]",
                        help="The learning rate (default: %(default)s)")
    parser.add_argument("--use-cuda", action="store_true", default=False,
                        help="Whether to use CUDA (GPU computation)")
    parser.add_argument("--stratified-sampler", action="store_true", default=False,
                        help="Draw samples with equal probabilities when training")
    parser.add_argument("--min-class-size", type=int, default=20, metavar="[INT]",
                        help="The minimum number of elements a class must has in the training set (default: %(default)s)")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Whether to show extra messages")
    parser.add_argument("--outdir", help="Path to output directory")
    parser.add_argument("--num-exp-genes", default=0.01, metavar="[FLOAT]", type=float,
                        help="The percentage of number of expressed genes (>= --min-gene-expression) a spot\n" \
                        "must have to be kept from the distribution of all expressed genes (0.0 - 1.0) (default: %(default)s)")
    parser.add_argument("--num-exp-spots", default=0.01, metavar="[FLOAT]", type=float,
                        help="The percentage of number of expressed spots a gene\n" \
                        "must have to be kept from the total number of spots (0.0 - 1.0) (default: %(default)s)")
    parser.add_argument("--min-gene-expression", default=1, type=int, metavar="[INT]", choices=range(1, 50),
                        help="The minimum count (number of reads) a gene must have in a spot to be\n"
                        "considered expressed (default: %(default)s)")
    args = parser.parse_args()
    main(args.train_data, args.test_data, args.train_classes, 
         args.test_classes, args.use_log_scale, args.normalization, 
         args.outdir, args.batch_correction, args.z_transformation, args.train_batch_size,
         args.test_batch_size, args.epochs, args.learning_rate, args.stratified_sampler, args.min_class_size, 
         args.use_cuda, args.num_exp_genes, args.num_exp_spots, args.min_gene_expression, args.verbose)