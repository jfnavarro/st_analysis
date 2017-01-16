""" 
Pre-processing functions for the ST Analysis packages.
Mainly function to aggregate datasets and filtering
functions (noisy spots and noisy genes)
"""
import numpy as np
import pandas as pd
import os
from stanalysis.normalization import *

def aggregate_datatasets(counts_table_files, plot_hist=False):
    """ This functions takes a list of data frames with ST data
    (genes as columns and spots as rows) and merges them into
    one data frame using the genes as merging criteria. 
    An index will append to each spot to be able to identify
    them. Optionally, a histogram of the read/spots and gene/spots
    distributions can be generated for each dataset.
    :param counts_table_files: a list of file names of the datasets
    :param plot_hist: True if we want to generate the histogram plots
    :return: a Pandas data frame with the merged data frames
    """
    # Spots are rows and genes are columns
    counts = pd.DataFrame()
    sample_counts = dict()
    for i,counts_file in enumerate(counts_table_files):
        if not os.path.isfile(counts_file):
            raise IOError("Error parsing data frame", "Invalid input file")
        new_counts = pd.read_table(counts_file, sep="\t", header=0, index_col=0)
        # Plot reads/genes distributions per spot
        if plot_hist:
            histogram(x_points=new_counts.sum(axis=1).values,
                      output=os.path.join(outdir, "hist_reads_{}.png".format(i)))
            histogram(x_points=(new_counts != 0).sum(axis=1).values, 
                      output=os.path.join(outdir, "hist_genes_{}.png".format(i)))
        # Append dataset index to the spots (indexes) so they can be traced
        new_spots = ["{0}_{1}".format(i, spot) for spot in new_counts.index]
        new_counts.index = new_spots
        counts = counts.append(new_counts)
    # Replace Nan and Inf by zeroes
    counts.replace([np.inf, -np.inf], np.nan)
    counts.fillna(0.0, inplace=True)
    return counts
  
def remove_noise(counts, num_exp_genes, num_exp_spots=1, min_expression=1):
    """This functions remove noisy genes and spots 
    for a given data frame (Genes as columns and spots as rows).
    - The noisy spots are removed so to keep a percentage
    of the total distribution of spots whose gene counts are not 0
    The percentage is given as a parameter.
    - The noisy genes are removed so every gene that is expressed
    in less than 1% of the total spots. Expressed with a count >= 2. 
    :param counts: a Pandas data frame with the counts
    :param num_exp_genes: an integer from 1-100 representing the % of 
    the distribution of expressed genes a spot must have to be kept
    :param num_exp_spots: a float from 0-100 representing the % of 
    the total number of spots that a gene must have with a count bigger
    than the parameter min_expression in order to be kept
    :param min_expression: the minimum expression for a gene to be
    considered expressed
    :return: a new Pandas data frame with noisy spots/genes removed
    """
    
    # How many spots do we keep based on the number of genes expressed?
    num_spots = len(counts.index)
    num_genes = len(counts.columns)
    min_genes_spot_exp = round((counts != 0).sum(axis=1).quantile(num_exp_genes))
    print "Number of expressed genes a spot must have to be kept " \
    "({}% of total expressed genes) {}".format(num_exp_genes,min_genes_spot_exp)
    counts = counts[(counts != 0).sum(axis=1) >= min_genes_spot_exp]
    print "Dropped {} spots".format(num_spots - len(counts.index))
          
    # Spots are columns and genes are rows
    counts = counts.transpose()
  
    # Remove noisy genes
    min_features_gene = round(len(counts.columns) * num_exp_spots) 
    print "Removing genes that are expressed in less than {} " \
    "spots with a count of at least {}".format(min_features_gene, min_expression)
    counts = counts[(counts >= MIN_EXPRESION).sum(axis=1) >= min_features_gene]
    print "Dropped {} genes".format(num_genes - len(counts.index))
    
    return counts.transpose()
    
def keep_top_genes(counts, num_genes_keep, criteria="Variance"):
    """ This function takes a Pandas data frame
    with ST data (Genes as columns and spots as rows)
    and returns a new data frame where only the top
    genes are kept by using the variance or the total count.
    :param counts: a Pandas data frame with the counts
    :param num_genes_keep: the % (1-100) of genes to keep
    :param criteria: the criteria used to select ("Variance or "TopRanked")
    :return: a new Pandas data frame with only the top raned genes. 
    """
    # Spots as columns and genes as rows
    counts = counts.transpose()
    # Keep only the genes with higher over-all variance
    num_genes = len(counts.index)
    if criteria == "Variance":
        min_genes_spot_var = counts.var(axis=1).quantile(num_genes_keep)
        print "Min normalized variance a gene must have over all spots " \
        "to be kept ({0}% of total) {1}".format(num_genes_keep, min_genes_spot_var)
        counts = counts[counts.var(axis=1) >= min_genes_spot_var]
    elif criteria == "TopRankded":
        min_genes_spot_sum = counts.sum(axis=1).quantile(num_genes_keep)
        print "Min normalized total count a gene must have over all spots " \
        "to be kept ({0}% of total) {1}".format(num_genes_keep, min_genes_spot_sum)
        counts = counts[counts.sum(axis=1) >= min_genes_spot_var]
    else:
        raise RunTimeError("Error, incorrect criteria method\n")  
    print "Dropped {} genes".format(num_genes - len(counts.index))    
    return counts.transpose()

def normalize_data(counts, normalization):
    """This functions takes a data frame as input
    with ST data (genes as columns and spots as rows) and 
    returns a data frame with the normalized counts using
    different methods.
    :param counts: a Pandas data frame with the counts
    :param normalization: the normalization method to use 
    (DESeq2, DESeq2Linear, DESeq2PseudoCount, DESeq2SizeAdjusted,RLE, REL, RAW, TMM, Scran)
    :return: a Pandas data frame with the normalized counts (genes as columns)
    """
    # Spots as columns and genes as rows
    counts = counts.transpose()
    # Per spot normalization
    if normalization in "DESeq2":
        size_factors = computeSizeFactors(counts)
    elif normalization in "DESeq2Linear":
        size_factors = computeSizeFactorsLinear(counts)
    elif normalization in "DESeq2PseudoCount":
        size_factors = computeSizeFactors(counts + 1)
    elif normalization in "DESeq2SizeAdjusted":
        size_factors = computeSizeFactorsSizeAdjusted(counts)
    elif normalization in "TMM":
        size_factors = computeTMMFactors(counts)
    elif normalization in "RLE":
        size_factors = computeRLEFactors(counts)
    elif normalization in "REL":
        size_factors = counts.sum(axis=0)
    elif normalization in "RAW":
        size_factors = 1
    elif normalization in "Scran":
        size_factors = computeSumFactors(counts)         
    else:
        raise RunTimeError("Error, incorrect normalization method\n")
    
    norm_counts = counts / size_factors   
    return counts.transpose()
    
def normalize_samples(counts, number_datasets):
    """ This function takes a data frame
    with ST data (genes as columns and spots as rows)
    that is composed by several datasets (the index
    of each dataset is appended to each spot) and
    then aggregates the counts for each gene
    in each dataset to later compute normalization
    factors for each dataset using DESeq. Finally
    it will apply the factors to each dataset. 
    :param counts: a Pandas dataframe conposed of several ST Datasets
    :param number_datasets: the number of different datasets merged in the input data frame
    :return: the same dataframe as input with the counts normalized
    """
    # First store the aggregated gene counts for each dataset in a dictionary
    sample_counts = dict()
    tot_spots = counts.index
    for i in xrange(number_datasets):
        spots_to_keep = [spot for spot in tot_spots if spot.startswith("{}_".format(i))]
        slice_counts = counts.loc[spots_to_keep]
        sample_counts[i] = slice_counts.sum(axis=0)
        
    # Now build up a data frame with the accumulated gene counts for
    # each sample
    per_sample_factors = pd.DataFrame(index=sample_counts.keys(), columns=counts.columns)
    for key,value in sample_counts.iteritems():
        per_sample_factors.loc[key] = value
    # Replace Nan and Inf by zeroes
    per_sample_factors.replace([np.inf, -np.inf], np.nan)
    per_sample_factors.fillna(0.0, inplace=True)
    
    # Spots are columns and genes are rows
    per_sample_factors = per_sample_factors.transpose()
    
    # Compute normalization factors for each dataset(sample) using DESeq 
    per_sample_size_factors = computeSizeFactors(per_sample_factors)
    
    # Now use the factors per sample to normalize genes in each sample
    # one factor per sample so we divide every gene count of each sample by its factor
    for spot in counts.index:
        # spot is i_XxY
        tokens = spot.split("x")
        assert(len(tokens) == 2)
        index = int(tokens[0].split("_")[0])
        factor = per_sample_size_factors[index]
        counts.loc[spot] = counts.loc[spot] / factor
        
    # Replace Nan and Inf by zeroes
    counts.replace([np.inf, -np.inf], np.nan)
    counts.fillna(0.0, inplace=True)
        
    return counts