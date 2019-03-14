import pandas as pd
import numpy as np

def split_dataset(dataset, labels, split_num=0.8, min_size=50):
    """Splits a dataset into two using the relative frequency
    given in split_num, the splitting will be performed by label
    so to ensure that all labels are kept and labels with less
    than min_size elements will be discarded
    """
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

def filter_classes(dataset, labels, min_size=50):
    """Given a dataset and a list of labels per spot (assuming same order)
    this function removes labels wit a minimum number
    of spots and return the filtered labels and dataset
    """
    train_indexes = list()
    train_labels = list()
    label_indexes = dict()
    # Store indexes for each cluster
    for i,label in enumerate(labels):
        try:
            label_indexes[label].append(i)
        except KeyError:
            label_indexes[label] = [i]
    # Keep only clusters bigger than min_size
    for label,indexes in label_indexes.items():
        if len(indexes) >= min_size:
            train_indexes += indexes
            train_labels += [label] * len(indexes)
    assert(len(train_labels) >= 0.1 * dataset.shape[0])
    # Return the reduced dataset/labels
    return dataset.iloc[train_indexes,:], train_labels

def update_labels(counts, labels_dict):
    """Given a dataset and a dictionary of spot -> labels
    This function creates a list of labels with the same
    order as in the dataset and removes the spots from
    the dataset that are not found in the dictionary
    """ 
    train_labels = list()
    for spot in counts.index:
        try:
            train_labels.append(labels_dict[spot])
        except KeyError:
            counts.drop(spot, axis=0, inplace=True)
    assert(len(train_labels) == counts.shape[0])
    return counts, train_labels   
    
def load_labels(filename):
    """Parses labels from a file
    The file must have two columns:
    SPOT LABEL
    """
    labels_dict = dict()
    with open(filename) as filehandler:
        for line in filehandler.readlines():
            tokens = line.split()
            assert(len(tokens) == 2)
            labels_dict[tokens[0]] = int(tokens[1])
    return labels_dict