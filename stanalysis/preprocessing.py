""" 
Pre-processing and normalization functions for the ST data.
"""

import numpy as np
import pandas as pd
import math
import os
from sklearn.preprocessing import StandardScaler
import re


def normalize(counts, normalization):
    """
    Wrapper around the function normalize_data()
    """
    return normalize_data(counts, normalization)


def filter_data_genes(counts, filter_genes):
    """
    Filter the input matrix of counts to keep only
    the genes given as input.
    :param counts: matrix of counts (genes as columns)
    :param filter_genes: list of genes to keep
    :return: the filtered matrix of counts
    """
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
        raise RuntimeError("No genes found in the datasets from the "
                           "list given\n{}\n".format(' '.join([x for x in filter_genes])))
    return counts.loc[:, genes_to_keep]


def filter_data(counts, num_exp_genes, num_exp_spots, min_gene_expression):
    """
    Filters the input matrix of counts with the thresholds given as input.
    :param counts: matrix of counts (genes as columns)
    :param num_exp_genes: the number of detected genes (>= min_gene_expression) a
    spot must have
    :param num_exp_spots: the number of detected spots (>= min_gene_expression) a
    gene must have
    :param min_gene_expression: expression value to define as detected
    :return: the filtered matrix of counts
    """
    if num_exp_spots <= 0.0 and num_exp_genes <= 0.0:
        return counts
    return remove_noise(counts, num_exp_genes, num_exp_spots,
                        min_expression=min_gene_expression)


def ztransformation(counts):
    """
    Applies a simple z-score transformation to
    a matrix of counts (genes as columns)
    which consists in substracting to each count 
    the mean of its column (gene) and then divide it by its
    the standard deviation
    :param counts: matrix of counts (genes as columns)
    :return: the z-scaled matrix of counts
    """
    scaler = StandardScaler()
    rows = counts.index
    cols = counts.columns
    scaled_counts = scaler.fit_transform(counts.values)
    return pd.DataFrame(data=scaled_counts,
                        index=rows,
                        columns=cols)


def aggregate_datatasets(counts_table_files, add_index=True, header=0):
    """
    Takes a list of matrices of counts (genes as columns and spots as rows)
    and merges them into one data frame using the genes as merging criteria.
    An index will be appended to each spot to be able to identify
    them (this is optional).
    :param counts_table_files: a list of file names corresponding to the matrices
    :param add_index: add the dataset index (position) to the spot's when True
    :return: a matrix counts with the merged data
    """
    # Spots are rows and genes are columns
    counts = pd.DataFrame()
    for i, counts_file in enumerate(counts_table_files):
        if not os.path.isfile(counts_file):
            raise IOError("Error parsing data frame", "Invalid input file")
        new_counts = pd.read_csv(counts_file, sep="\t",
                                 header=header, index_col=0, engine='c', low_memory=True)
        new_counts = new_counts[~new_counts.index.duplicated()]
        # Append dataset index to the spots (indexes) so they can be traced
        if add_index and len(counts_table_files) > 1:
            new_spots = ["{0}_{1}".format(i + 1, spot) for spot in new_counts.index]
            new_counts.index = new_spots
        counts = counts.append(new_counts, sort=True)
    # Replace Nan and Inf by zeroes
    counts.replace([np.inf, -np.inf], np.nan)
    counts.fillna(0.0, inplace=True)
    return counts


def remove_noise(counts, num_exp_genes=0.01, num_exp_spots=0.01, min_expression=1):
    """
    This functions remove noisy (low qualityh) genes and spots
    for a given matrix of counts (Genes as columns and spots as rows).
    - The noisy spots are removed so to keep a percentage
    of the total distribution of spots whose gene counts >= min_expression
    The percentage is given as a parameter (0.0 - 1.0).
    - The noisy genes are removed so every gene that is expressed
    in less than a percentage of the total spots whose gene counts >= min_expression
    The percentage is given as a parameter (0.0 - 1.0).
    :param counts: a matrix of counts
    :param num_exp_genes: a float from 0-1 representing the % of 
    the distribution of expressed genes a spot must have to be kept
    :param num_exp_spots: a float from 0-1 representing the % of 
    the total number of spots that a gene must have with a count bigger
    than the parameter min_expression in order to be kept
    :param min_expression: the minimum expression for a gene to be
    considered expressed
    :return: a new matrix of counts with noisy spots/genes removed
    """

    # How many spots do we keep based on the number of genes expressed?
    num_spots = len(counts.index)
    num_genes = len(counts.columns)
    if 0.0 < num_exp_genes < 1.0:
        # Remove noisy spots
        gene_sums = (counts >= min_expression).sum(axis=1)
        min_genes_spot_exp = round(gene_sums.quantile(num_exp_genes))
        print("Number of expressed genes (count of at least {}) a spot must have to be kept "
              "({}% of total expressed genes) {}".format(min_expression, num_exp_genes, min_genes_spot_exp))
        counts = counts[gene_sums >= min_genes_spot_exp]
        print("Dropped {} spots".format(num_spots - len(counts.index)))
    if 0.0 < num_exp_spots < 1.0:
        # Spots are columns and genes are rows
        counts = counts.transpose()
        # Remove noisy genes
        min_features_gene = round(len(counts.columns) * num_exp_spots)
        print("Removing genes that are expressed in less than {} "
              "spots with a count of at least {}".format(min_features_gene, min_expression))
        counts = counts[(counts >= min_expression).sum(axis=1) >= min_features_gene]
        print("Dropped {} genes".format(num_genes - len(counts.index)))
        counts = counts.transpose()
    return counts


def keep_top_genes(counts, num_genes_discard, criteria="Variance"):
    """
    This function takes a matrix of counts (Genes as columns and spots as rows)
    and returns a new matrix of counts where a number of genesa re kept
    using the variance or the total count as filtering criterias.
    :param counts: a matrix of counts
    :param num_genes_discard: the % (1-100) of genes to keep
    :param criteria: the criteria used to select ("Variance or "TopRanked")
    :return: a new matrix of counts with only the top ranked genes.
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
            print("Computed variance is NaN! Check your input data.")
        else:
            print("Min normalized variance a gene must have over all spots "
                  "to be kept ({0}% of total) {1}".format(num_genes_discard, min_genes_spot_var))
            counts = counts[var >= min_genes_spot_var]
    elif criteria == "TopRanked":
        sum = counts.sum(axis=1)
        min_genes_spot_sum = sum.quantile(num_genes_discard)
        if math.isnan(min_genes_spot_sum):
            print("Computed sum is NaN! Check your input data.")
        else:
            print("Min normalized total count a gene must have over all spots "
                  "to be kept ({0}% of total) {1}".format(num_genes_discard, min_genes_spot_sum))
            counts = counts[sum >= min_genes_spot_sum]
    else:
        raise RuntimeError("Error, incorrect criteria method\n")
    print("Dropped {} genes".format(num_genes - len(counts.index)))
    return counts.transpose()


def compute_size_factors(counts, normalization):
    """
    Helper function to compute normalization size factors
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
        raise RuntimeError("Error, incorrect normalization method\n")
    return size_factors


def normalize_data(counts, normalization):
    """
    This functions takes a matrix of counts as input
    (genes as columns and spots as rows) and
    returns a new matrix of counts normalized using
    the normalization method given in the input.
    :param counts: a matrix of counts (genes as columns)
    :param normalization: the normalization method to use (RAW, REL or CPM)
    :return: a matrix of counts with normalized counts (genes as columns)
    """
    # Spots as columns and genes as rows
    norm_counts = counts.transpose()
    if normalization in "REL":
        norm_counts = norm_counts / norm_counts.sum(axis=1)
    elif normalization in "CPM":
        col_sums = counts.sum(axis=1)
        norm_counts = (norm_counts / col_sums) * np.mean(col_sums)
    elif normalization in "RAW":
        pass
    # return normalize counts (genes as columns)
    return norm_counts.transpose()
