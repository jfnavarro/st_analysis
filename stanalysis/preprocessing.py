""" 
Pre-processing functions for the ST Analysis packages.
Mainly function to aggregate datasets and filtering
functions (noisy spots and noisy genes)
"""
import numpy as np
import pandas as pd
import math
import os
from sklearn.preprocessing import StandardScaler
import re

def normalize(counts, normalization):
    return normalize_data(counts,
                          normalization,
                          center=False)

def filter_data_genes(counts, filter_genes):
    # Extract the list of the genes that must be shown
    genes_to_keep = list()
    if filter_genes:
        for gene in counts.columns:
            for regex in filter_genes:
                if re.fullmatch(regex, gene):
                    genes_to_keep.append(gene)
                    break                         
    else: 
        genes_to_keep = counts.columns
    # Check that we hit some genes
    if len(genes_to_keep) == 0:
        raise RuntimeError("No genes found in the datasets from the " \
                         "list given\n{}\n".format(' '.join([x for x in filter_genes])))
    counts = counts.loc[:,genes_to_keep]
    return counts

def filter_data(counts, num_exp_genes, num_exp_spots, min_gene_expression):
    if num_exp_spots <= 0.0 and num_exp_genes <= 0.0:
        return counts
    return remove_noise(counts, num_exp_genes, num_exp_spots,
                        min_expression=min_gene_expression)
    
def ztransformation(counts):
    """ Applies a simple z-score transformation to 
    a ST data frame (genes as columns)
    which consists in substracting to each count 
    the mean of its column (gene) and then divide it by its
    the standard deviation
    """
    scaler = StandardScaler()
    rows = counts.index
    cols = counts.columns
    scaled_counts = scaler.fit_transform(counts.values)
    return pd.DataFrame(data=scaled_counts,
                        index=rows,
                        columns=cols)
    
def aggregate_datatasets(counts_table_files, add_index=True):
    """ This functions takes a list of data frames with ST data
    (genes as columns and spots as rows) and merges them into
    one data frame using the genes as merging criteria. 
    An index will be appended to each spot to be able to identify
    them (optional).
    :param counts_table_files: a list of file names of the datasets
    :param add_index: add the dataset index to the spot's
    :return: a Pandas data frame with the merged data frames
    """
    if len(counts_table_files) == 1:
        return pd.read_csv(counts_table_files[0], sep="\t", 
                           header=0, index_col=0, engine='c', low_memory=True)
    # Spots are rows and genes are columns
    counts = pd.DataFrame()
    for i,counts_file in enumerate(counts_table_files):
        if not os.path.isfile(counts_file):
            raise IOError("Error parsing data frame", "Invalid input file")
        new_counts = pd.read_csv(counts_file, sep="\t", 
                                 header=0, index_col=0, engine='c', low_memory=True)
        # Append dataset index to the spots (indexes) so they can be traced
        if add_index:
            new_spots = ["{0}_{1}".format(i + 1, spot) for spot in new_counts.index]
            new_counts.index = new_spots
        counts = counts.append(new_counts, sort=True)
    # Replace Nan and Inf by zeroes
    counts.replace([np.inf, -np.inf], np.nan)
    counts.fillna(0.0, inplace=True)
    return counts
  
def remove_noise(counts, num_exp_genes=0.01, num_exp_spots=0.01, min_expression=1):
    """This functions remove noisy genes and spots 
    for a given ST data frame (Genes as columns and spots as rows).
    - The noisy spots are removed so to keep a percentage
    of the total distribution of spots whose gene counts >= min_expression
    The percentage is given as a parameter (0.0-1.0).
    - The noisy genes are removed so every gene that is expressed
    in less than a percentage of the total spots whose gene counts >= min_expression
    The percentage is given as a parameter (0.0-1.0).
    :param counts: a data frame with the counts
    :param num_exp_genes: a float from 0-1 representing the % of 
    the distribution of expressed genes a spot must have to be kept
    :param num_exp_spots: a float from 0-1 representing the % of 
    the total number of spots that a gene must have with a count bigger
    than the parameter min_expression in order to be kept
    :param min_expression: the minimum expression for a gene to be
    considered expressed
    :return: a new data frame with noisy spots/genes removed
    """
    
    # How many spots do we keep based on the number of genes expressed?
    num_spots = len(counts.index)
    num_genes = len(counts.columns)
    
    if num_exp_genes > 0.0 and num_exp_genes < 1.0:
        gene_sums = (counts >= min_expression).sum(axis=1)
        min_genes_spot_exp = round(gene_sums.quantile(num_exp_genes))
        print("Number of expressed genes (count of at least {}) a spot must have to be kept " \
        "({}% of total expressed genes) {}".format(min_expression, num_exp_genes, min_genes_spot_exp))
        counts = counts[gene_sums >= min_genes_spot_exp]
        print("Dropped {} spots".format(num_spots - len(counts.index)))
        
    if num_exp_spots > 0.0 and num_exp_spots < 1.0:  
        # Spots are columns and genes are rows
        counts = counts.transpose()
        # Remove noisy genes
        min_features_gene = round(len(counts.columns) * num_exp_spots) 
        print("Removing genes that are expressed in less than {} " \
        "spots with a count of at least {}".format(min_features_gene, min_expression))
        counts = counts[(counts >= min_expression).sum(axis=1) >= min_features_gene]
        print("Dropped {} genes".format(num_genes - len(counts.index)))
        counts = counts.transpose()
    
    return counts 
    
def keep_top_genes(counts, num_genes_discard, criteria="Variance"):
    """ This function takes a data frame
    with ST data (Genes as columns and spots as rows)
    and returns a new data frame where only the top
    genes are kept by using the variance or the total count.
    :param counts: a data frame with the counts
    :param num_genes_discard: the % (1-100) of genes to keep
    :param criteria: the criteria used to select ("Variance or "TopRanked")
    :return: a new data frame with only the top ranked genes. 
    """
    if num_genes_discard <= 0:
        return counts
    # Spots as columns and genes as rows
    counts = counts.transpose()
    # Keep only the genes with higher over-all variance
    num_genes = len(counts.index)
    print("Removing {}% of genes based on the {}".format(num_genes_discard * 100, criteria))
    if criteria == "Variance":
        var = counts.var(axis=1)
        min_genes_spot_var = var.quantile(num_genes_discard)
        if math.isnan(min_genes_spot_var):
            print("Computed variance is NaN! Check your normalization factors..")
        else:
            print("Min normalized variance a gene must have over all spots " \
            "to be kept ({0}% of total) {1}".format(num_genes_discard, min_genes_spot_var))
            counts = counts[var >= min_genes_spot_var]
    elif criteria == "TopRanked":
        sum = counts.sum(axis=1)
        min_genes_spot_sum = sum.quantile(num_genes_discard)
        if math.isnan(min_genes_spot_var):
            print("Computed sum is NaN! Check your normalization factors..")
        else:
            print("Min normalized total count a gene must have over all spots " \
            "to be kept ({0}% of total) {1}".format(num_genes_discard, min_genes_spot_sum))
            counts = counts[sum >= min_genes_spot_var]
    else:
        raise RunTimeError("Error, incorrect criteria method\n")  
    print("Dropped {} genes".format(num_genes - len(counts.index)))
    return counts.transpose()

def compute_size_factors(counts, normalization):
    """ Helper function to compute normalization size factors
    """
    counts = counts.transpose()
    if normalization in "REL":
        size_factors = counts.sum(axis=0)
    elif normalization in "CPM":
        col_sums = counts.sum(axis=0)
        size_factors = col_sums * np.mean(col_sums)
    elif normalization in "RAW":
        size_factors = 1
    else:
        raise RunTimeError("Error, incorrect normalization method\n")   
    return size_factors

def normalize_data(counts, normalization, center=False):
    """This functions takes a data frame as input
    with ST data (genes as columns and spots as rows) and 
    returns a data frame with the normalized counts using
    different methods.
    :param counts: a data frame with the counts
    :param normalization: the normalization method to use (RAW, REL or CPM)
    :param center: if True the size factors will be centered by their mean
    :return: a Pandas data frame with the normalized counts (genes as columns)
    """
    # Compute the size factors
    size_factors = compute_size_factors(counts, normalization)
    if np.all(size_factors == 1.0):
        return counts
    if np.isnan(size_factors).any() or np.isinf(size_factors).any() \
    or np.any(size_factors <= 0.0):
        print("Warning: Computed size factors contained NaN or zeroes or Inf values."
              "\nSpots for these will be discarded!")
        valid = (size_factors > 0) & np.isfinite(size_factors)
        counts = counts[valid]
        size_factors = size_factors[valid]
    # Spots as columns and genes as rows
    counts = counts.transpose()
    # Center size factors if requested
    if center: 
        size_factors = size_factors - np.mean(size_factors)
    norm_counts = counts / size_factors
    # return normalize counts (genes as columns)
    return norm_counts.transpose()