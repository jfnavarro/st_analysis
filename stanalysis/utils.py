import pandas as pd
import numpy as np

def parse_label(filename, min_size):
    labels = pd.read_table(filename, sep="\t", header=None, index_col=0)
    labels.columns = ["cluster"]
    unique_elements, counts_elements = np.unique(labels["cluster"], return_counts=True)
    return labels[labels["cluster"].isin(unique_elements[counts_elements > min_size])]
    
def split_dataset(dataset, labels, vali_split, test_split, min_size=50):
    """Splits a dataset into three using the relative frequency
    given in vali_size and test_size. The splitting will be performed by label
    so to ensure that all labels are kept and labels with less
    than min_size elements will be discarded
    """
    train_indexes = list()
    vali_indexes = list()
    test_indexes = list()
    train_labels = list()
    vali_labels = list()
    test_labels = list()
    label_indexes = dict()
    # Store indexes for each cluster
    for i,label in enumerate(labels):
        try:
            label_indexes[label].append(i)
        except KeyError:
            label_indexes[label] = [i]
    # Split randomly indexes for each cluster into training, validation and testing sets  
    for label,indexes in label_indexes.items():
        if len(indexes) >= min_size:
            indexes = np.asarray(indexes)
            np.random.shuffle(indexes) 
            n_ele_vali = int(vali_split * len(indexes))
            n_ele_test = int(test_split * len(indexes))
            n_ele_train = len(indexes) - (n_ele_vali + n_ele_test)
            training = indexes[:n_ele_train]
            validation = indexes[n_ele_train:n_ele_train + n_ele_vali]
            testing = indexes[n_ele_train + n_ele_vali:n_ele_train + n_ele_vali + n_ele_test]
            train_indexes += training.tolist()
            vali_indexes += validation.tolist()
            test_indexes += testing.tolist()
            train_labels += [label] * len(training)
            vali_labels += [label] * len(validation)
            test_labels += [label] * len(testing)         
    # Return the split datasets and their labels
    return dataset.iloc[train_indexes,:], dataset.iloc[vali_indexes,:], dataset.iloc[test_indexes,:], train_labels, vali_labels, test_labels

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