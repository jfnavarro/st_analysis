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
import gc

from collections import defaultdict

from stanalysis.preprocessing import *

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as utils
import torchvision

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit

import gc

# Windows work-around
__spec__ = None
import multiprocessing

SEARCH_BATCH = [(100,100), (200,200), (500,500), (1000, 1000), (2000, 1000), (3000, 1000), (4000,1000)]
SEARCH_LR = [0.1,0.01,0.05,0.001,0.005,0.0001,0.0005,0.00001]
SEARCH_HL = [(3000,500), (2000,500), (1000,500), (3000, 1000), (2000, 1000), (2000, 300), (1000,300), (500,300)]

def computeWeightsClasses(dataset):
    # Distribution of labels
    label_count = defaultdict(int)
    for _,label in dataset:
        label_count[label.item()] += 1         
    # Weight for each sample
    weights = np.asarray([1.0 / x for x in label_count.values()])
    return weights

def computeWeights(dataset, nclasses):
    count = [0] * nclasses                                                      
    for item in dataset:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(dataset)                                              
    for idx, val in enumerate(dataset):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return np.asarray(weight)  
    
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

def create_model(n_feature, n_class, hidden_layer_one, hidden_layer_two, af):
    # Init model
    H1 = hidden_layer_one
    H2 = hidden_layer_two
    model = torch.nn.Sequential(
        torch.nn.Linear(n_feature, H1),
        torch.nn.BatchNorm1d(num_features=H1),
        torch.nn.Tanh() if af == "tanh" else torch.nn.SELU(),
        torch.nn.Linear(H1, H2),
        torch.nn.BatchNorm1d(num_features=H2),
        torch.nn.Tanh() if af == "tanh" else torch.nn.SELU(),
        torch.nn.Linear(H2, n_class),
    )
    return model   
   
def create_loaders(trn_set, vali_set, 
                   train_batch_size, validation_batch_size, 
                   train_sampler, test_sampler, 
                   shuffle_train, shuffle_test, 
                   kwargs): 
    # Create loaders
    trn_loader = utils.DataLoader(trn_set, 
                                  sampler=train_sampler, 
                                  shuffle=shuffle_train,
                                  batch_size=train_batch_size, 
                                  **kwargs)
    vali_loader = utils.DataLoader(vali_set, 
                                  sampler=test_sampler, 
                                  shuffle=shuffle_test,
                                  batch_size=validation_batch_size, 
                                  **kwargs)
    
    return trn_loader, vali_loader
    
def train(model, trn_loader, optimizer, loss_func, device):
    model.train()
    training_loss = 0
    training_acc = 0
    for data, target in trn_loader:
        data = Variable(data.to(device))
        target = Variable(target.to(device))
        # Forward pass
        output = model(data)
        tloss = loss_func(output, target)
        training_loss += tloss.item()
        # Zero the gradients
        optimizer.zero_grad()
        # Backward pass
        tloss.backward()
        # Update parameters
        optimizer.step()
        # Compute prediction's score
        pred = torch.argmax(output.data, 1)
        training_acc += accuracy_score(target.data.cpu().numpy(),
                                       pred.data.cpu().numpy())
    avg_loss = training_loss / float(len(trn_loader.dataset))
    avg_acc = training_acc / float(len(trn_loader.dataset))
    return avg_loss, avg_acc
        
def test(model, vali_loader, loss_func, device):
    model.eval()
    test_loss = 0
    preds = list()
    for data, target in vali_loader:
        with torch.no_grad():
            data = Variable(data.to(device))
            target = Variable(target.to(device))
            output = model(data)
            test_loss += loss_func(output, target).item()
            pred = torch.argmax(output.data, 1)
            preds += pred.cpu().numpy().tolist()
    avg_loss = test_loss / float(len(vali_loader.dataset))  
    return preds, avg_loss

def predict(model, data, device):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        output = model(data)
        pred = torch.argmax(output.data, 1)
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
         log_scale, 
         normalization,
         stratified_loss,
         outdir, 
         batch_correction,
         standard_transformation,
         train_batch_size,
         validation_batch_size, 
         epochs, 
         learning_rate,
         stratified_sampler,
         min_class_size,
         use_cuda,
         num_exp_genes, 
         num_exp_spots,
         min_gene_expression,
         verbose,
         hidden_layer_one, 
         hidden_layer_two, 
         train_validation_ratio,
         grid_search):

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
       
    if normalization == "Scran" and log_scale:
        sys.stderr.write("Warning, when performing Scran normalization log-scale will be ignored\n")
     
    if batch_correction and log_scale:
        sys.stderr.write("Warning, when performing batch correction log-scale will be ignored\n")
                  
    if min_class_size < 0:
        sys.stderr.write("Error, invalid minimum class size\n")
        sys.exit(1)

    if learning_rate < 0:
        sys.stderr.write("Error, invalid learning rate\n")
        sys.exit(1)
        
    if hidden_layer_one <= 0 or hidden_layer_two <= 0:
        sys.stderr.write("Error, invalid hidden layers\n")
        sys.exit(1)
        
    if train_batch_size < 5 or validation_batch_size < 5:
        sys.stderr.write("Error, batch size is too small\n")
        sys.exit(1)
        
    if epochs < 5:
        sys.stderr.write("Error, number of epoch is too small\n")
        sys.exit(1)
    
    if num_exp_genes < 0.0 or num_exp_genes > 1.0:
        sys.stderr.write("Error, invalid number of expressed genes\n")
        sys.exit(1)
        
    if num_exp_spots < 0.0 or num_exp_spots > 1.0:
        sys.stderr.write("Error, invalid number of expressed genes\n")
        sys.exit(1)

    if train_validation_ratio < 0.2 or train_validation_ratio > 1.0:
        sys.stderr.write("Error, invalid train test ratio genes\n")
        sys.exit(1)
         
    if not torch.cuda.is_available() and use_cuda:
        sys.stderr.write("Error, CUDA is not available in this computer\n")
        sys.exit(1)
          
    if not outdir or not os.path.isdir(outdir):
        outdir = os.getcwd()   
    print("Output folder {}".format(outdir))
    
    print("Loading training dataset...")
    train_data_frame = pd.read_table(train_data, sep="\t", header=0, index_col=0, 
                                     engine='c', low_memory=True)
    # Remove noisy genes
    train_data_frame = remove_noise(train_data_frame, 1.0, num_exp_spots, min_gene_expression)
    train_genes = list(train_data_frame.columns.values)
    
    # Load all the classes for the training set
    train_labels_dict = load_labels(train_classes_file)
    train_data_frame, train_labels = update_labels(train_data_frame, train_labels_dict)
    
    print("Loading testing dataset...")
    test_data_frame = pd.read_table(test_data, sep="\t", header=0, index_col=0,
                                    engine='c', low_memory=True)
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
    print("Genes in testing set {}".format(test_data_frame.shape[1]))
    print("Spots in testing set {}".format(test_data_frame.shape[0]))
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
        train_data_frame.to_csv("train_bc_final.tsv", sep="\t")
        test_data_frame.to_csv("test_bc_final.tsv", sep="\t")
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
    print("Splitting training set into training and validation sets (equally balancing clusters)\n"\
          "with a ratio of {} and discarding classes with less than {} elements".format(train_validation_ratio, min_class_size))
    train_counts_x, train_counts_y, train_labels_x, train_labels_y = split_dataset(train_data_frame, 
                                                                                   train_labels, 
                                                                                   train_validation_ratio, 
                                                                                   min_class_size)
    
    # Update the maps of indexes to labels again since some labels may have been removed
    #TODO this is ugly, a better approach should be implemented
    labels_index_map_filtered = dict()
    index_label_map_filtered = dict()
    for i,label in enumerate(sorted(set(train_labels_x))):
        labels_index_map_filtered[label] = i
        index_label_map_filtered[i] = index_label_map[label]
    print("Mapping of labels (filtered):")
    print(index_label_map_filtered)
    train_labels_x = [labels_index_map_filtered[x] for x in train_labels_x]
    train_labels_y = [labels_index_map_filtered[x] for x in train_labels_y] 
    
    print("Training set {}".format(train_counts_x.shape[0]))
    print("Validation set {}".format(train_counts_y.shape[0]))
    
    # PyTorch needs floats
    train_counts = train_counts_x.astype(np.float32).values
    vali_counts = train_counts_y.astype(np.float32).values
    del train_counts_x
    del train_counts_y
    del train_data_frame
    gc.collect()
    
    # Input and output sizes
    n_feature = train_counts.shape[1]
    n_ele_train = train_counts.shape[0]
    n_ele_test = vali_counts.shape[0]
    n_class = max(set(train_labels_x)) + 1
    
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    
    if not use_cuda:
        workers = 4
        print("Workers {}".format(workers))
        kwargs = {'num_workers': workers, 'pin_memory': False}
    else:
        kwargs = {'num_workers': 1, 'pin_memory': True}

    # Create Tensor Flow train dataset
    X_train = torch.tensor(train_counts)
    X_vali = torch.tensor(vali_counts)
    y_train = torch.from_numpy(np.asarray(train_labels_x, dtype=np.longlong))
    y_vali = torch.from_numpy(np.asarray(train_labels_y, dtype=np.longlong))
    del train_counts
    del vali_counts
    gc.collect()
    
    # Create tensor datasets (train + test)
    trn_set = utils.TensorDataset(X_train, y_train)
    vali_set = utils.TensorDataset(X_vali, y_vali)
        
    if stratified_loss:
        print("Using a stratified loss...")
        # Compute weights
        weights_classes = computeWeightsClasses(trn_set)
        weights_classes = torch.from_numpy(weights_classes).float().to(device)
    else:
        weights_classes = None  

    # Creating loss
    loss_func = torch.nn.CrossEntropyLoss(weight=weights_classes, reduction="sum")
    
    # Create Samplers
    if stratified_sampler:
        print("Using a stratified sampler for training set...")
        weights_train = computeWeights(trn_set, n_class)
        weights_train = torch.from_numpy(weights_train).float().to(device)
        trn_sampler = utils.sampler.WeightedRandomSampler(weights_train, 
                                                          len(weights_train), 
                                                          replacement=False) 
        weights_vali = computeWeights(vali_set, n_class)
        weights_vali = torch.from_numpy(weights_vali).float().to(device)
        vali_sampler = utils.sampler.WeightedRandomSampler(weights_vali, 
                                                           len(weights_vali), 
                                                           replacement=False) 
    else:
        trn_sampler = None   
        vali_sampler = None
    
    learning_rates = [learning_rate] if not grid_search else SEARCH_LR
    batch_sizes = [(train_batch_size, validation_batch_size)] if not grid_search else SEARCH_BATCH
    hidden_sizes = [(2000, 1000)] if not grid_search else SEARCH_HL
    best_model = dict()
    best_acc = 0
    best_lr = 0
    best_bs = (0,0)
    best_h = (0,0)
    for lr in learning_rates:
        for (trn_bs, vali_bs) in batch_sizes:
            for (h1, h2) in hidden_sizes:
                # Create model
                model = create_model(n_feature, n_class, h1, h2, af="tanh")
                model = model.to(device)
                # Create optimizer
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
                # Create loaders
                trn_loader, vali_loader = create_loaders(trn_set, vali_set, 
                                                         trn_bs, vali_bs,
                                                         trn_sampler, vali_sampler, 
                                                         not stratified_sampler, 
                                                         not stratified_sampler,
                                                         kwargs)
                # Train the model
                best_local_loss = 10e6
                best_local_acc = 0
                best_epoch_idx = 0
                counter = 0
                history = list()
                best_model_local = dict()
                if grid_search:
                    print("Training model with:\n learning rate {}\n train batch size {}\n "\
                          "test batch size {}\n hidden layer one {}\n hidden layer two {}\n".format(lr,trn_bs,vali_bs,h1,h2))
                for epoch in range(epochs):
                    if verbose:
                        print('Epoch: {}'.format(epoch))
                        
                    # Training
                    avg_train_loss, avg_training_acc = train(model, trn_loader, optimizer, loss_func, device)
                    history.append((avg_training_acc, avg_train_loss))

                    if verbose:
                        print("Training set accuracy {}".format(avg_training_acc))
                        print("Training set loss (avg) {}".format(avg_train_loss))
                        print("Testing set accuracy {}".format(avg_testing_acc))
                        print("Testing set loss (avg) {}".format(avg_test_loss))
                       
                    # Check if the accuracy got better
                    if avg_training_acc > best_local_acc:
                        best_local_acc = avg_training_acc 
                        best_epoch_idx = epoch
                        best_model_local = model.state_dict()
                        
                    # Check if the loss is not improving
                    if avg_train_loss < best_local_loss:
                        counter = 0
                        best_local_loss = avg_train_loss
                    else:
                        counter += 1
                    
                    # Early out
                    if counter >= 20:
                        if verbose: 
                            print("Early stopping at epoch {}".format(epoch))
                        break
                       
                # Test the model on the validation set
                model.load_state_dict(best_model_local)   
                preds, avg_test_loss = test(model, vali_loader, loss_func, device)

                # Compute accuracy score
                avg_testing_acc = accuracy_score(y_vali.cpu().numpy(), preds)
                      
                # Check the results to keep the best model
                avg_train_acc, avg_train_loss = history[best_epoch_idx]
                print("Best training accuracy {} and loss (avg.) {}".format(avg_train_acc, avg_train_loss))
                print("Best testing accuracy {} and loss (avg.) {}".format(avg_testing_acc, avg_test_loss))
                if avg_testing_acc > best_acc:
                    best_acc = avg_testing_acc
                    best_model = best_model_local
                    best_lr = lr
                    best_bs = (trn_bs, vali_bs)
                    best_h = (h1,h2)

    print("Model trained!")
    print("Learning rate {}".format(best_lr))
    print("Train batch size {}".format(best_bs[0]))
    print("Validation batch size {}".format(best_bs[1]))
    print("Hidden layer one {}".format(best_h[0]))
    print("Hidden layer two {}".format(best_h[1]))
    print("Accuracy score {}".format(best_acc))
    
    # Load and save best model
    model = create_model(n_feature, n_class, best_h[0], best_h[1], af="tanh")
    model.load_state_dict(best_model)
    torch.save(model, os.path.join(outdir, "model.pt"))
        
    # Predict
    print("Predicting on test data..")
    predict_counts = test_data_frame.astype(np.float32).values
    test_index = test_data_frame.index
    del test_data_frame
    gc.collect()
    
    X_pre = torch.tensor(predict_counts)
    y_pre = test_labels if test_classes_file is not None else None
    out, preds = predict(model, X_pre, device)
    # Map labels back to their original value
    preds = [index_label_map_filtered[np.asscalar(x)] for x in preds.cpu().numpy()]
    if y_pre is not None:
        print("Classification report\n{}".
              format(classification_report(y_pre, preds)))
        print("Confusion matrix:\n{}".format(confusion_matrix(y_pre, preds)))
    with open(os.path.join(outdir, "predicted_classes.tsv"), "w") as filehandler:
        for spot, pred, probs in zip(test_index, preds, out.cpu().numpy()):
            filehandler.write("{0}\t{1}\t{2}\n".format(spot, pred,
                                                       "\t".join(['{:.6f}'.format(x) for x in probs]))) 
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--train-data", required=True, type=str,
                        help="Path to the input training dataset (matrix of counts, spots as rows)")
    parser.add_argument("--test-data", required=True, type=str,
                        help="Path to the test dataset (to be predicted) (matrix of counts, spots as rows)")
    parser.add_argument("--train-classes", required=True, type=str,
                        help="Path to the training classes file (SPOT LABEL)")
    parser.add_argument("--test-classes", required=False, type=str,
                        help="Path to the test classes file (SPOT LABEL)")
    parser.add_argument("--log-scale", action="store_true", default=False,
                        help="Convert the training and test sets to log space (not applied when using batch correction)")
    parser.add_argument("--batch-correction", action="store_true", default=False,
                        help="Perform batch-correction (Scran::Mnncorrect()) between train and test sets")
    parser.add_argument("--standard-transformation", action="store_true", default=False,
                        help="Apply the standard transformation to each gene on the train and test sets")
    parser.add_argument("--normalization", default="RAW", metavar="[STR]", 
                        type=str, 
                        choices=["RAW", "DESeq2",  "REL", "Scran"],
                        help="Normalize the counts using:\n" \
                        "RAW = absolute counts\n" \
                        "DESeq2 = DESeq2::estimateSizeFactors()\n" \
                        "Scran = Deconvolution Sum Factors (Marioni et al)\n" \
                        "REL = Each gene count divided by the total count of its spot\n" \
                        "(default: %(default)s)")
    parser.add_argument("--train-batch-size", type=int, default=1000, metavar="[INT]",
                        help="The input batch size for training (default: %(default)s)")
    parser.add_argument("--validation-batch-size", type=int, default=1000, metavar="[INT]",
                        help="The input batch size for validation (default: %(default)s)")
    parser.add_argument("--epochs", type=int, default=50, metavar="[INT]",
                        help="The number of epochs to train (default: %(default)s)")
    parser.add_argument("--hidden-layer-one", type=int, default=2000, metavar="[INT]",
                        help="The number of neurons in the first hidden layer (default: %(default)s)")
    parser.add_argument("--hidden-layer-two", type=int, default=1000, metavar="[INT]",
                        help="The number of neurons in the second hidden layer (default: %(default)s)")
    parser.add_argument("--train-validation-ratio", type=float, default=0.8, metavar="[FLOAT]",
                        help="The percentage of the training set that will be used to validate"\
                        " the model during training (default: %(default)s)")
    parser.add_argument("--learning-rate", type=float, default=0.001, metavar="[FLOAT]",
                        help="The learning rate (default: %(default)s)")
    parser.add_argument("--use-cuda", action="store_true", default=False,
                        help="Whether to use CUDA (GPU computation)")
    parser.add_argument("--stratified-sampler", action="store_true", default=False,
                        help="Draw samples with equal probabilities when training")
    parser.add_argument("--stratified-loss", action="store_true", default=False,
                        help="Penalizes more small classes in the loss")
    parser.add_argument("--min-class-size", type=int, default=20, metavar="[INT]",
                        help="The minimum number of elements a class must has in the" \
                        " training set (default: %(default)s)")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Whether to show extra messages")
    parser.add_argument("--grid-search", action="store_true", default=False,
                        help="Perform a grid search to find the most optimal parameters")
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
         args.test_classes, args.log_scale, args.normalization, 
         args.stratified_loss, args.outdir, args.batch_correction, 
         args.standard_transformation, args.train_batch_size, args.validation_batch_size, 
         args.epochs, args.learning_rate, args.stratified_sampler, 
         args.min_class_size, args.use_cuda, args.num_exp_genes, 
         args.num_exp_spots, args.min_gene_expression, args.verbose,
         args.hidden_layer_one, args.hidden_layer_two, args.train_validation_ratio, 
         args.grid_search)