#! /usr/bin/env python
""" 
This script performs Differential Expression Analysis 
using DESeq2 or Scran + DESeq2 on ST datasets.

The script can take one or several datasets with the following format:

      GeneA   GeneB   GeneC
1x1   
1x2
...

Ideally, each dataset (matrix) would correspond to a region
of interest (Selection) to be compared. 

The script also needs the list of comparisons to make (1 vs 2, etc..)
Each comparison will be performed between datasets and the input should be:

DATASET-DATASET DATASET-DATASET ...

The script will output the list of up-regulated and down-regulated genes
for each possible DEA comparison (between tables) as well as a set of volcano plots.

NOTE: soon Monocle and edgeR will be added 

@Author Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>
"""
import argparse
import sys
import os
import numpy as np
import pandas as pd
from stanalysis.normalization import RimportLibrary
from stanalysis.preprocessing import compute_size_factors, aggregate_datatasets, remove_noise
from stanalysis.visualization import volcano
from stanalysis.analysis import deaDESeq2, deaScranDESeq2
import matplotlib.pyplot as plt
    
def main(counts_table_files, conditions, comparisons, outdir, fdr, 
         normalization, num_exp_spots, num_exp_genes, min_gene_expression):

    if len(counts_table_files) == 0 or \
    any([not os.path.isfile(f) for f in counts_table_files]):
        sys.stderr.write("Error, input file/s not present or invalid format\n")
        sys.exit(1)
     
    if not outdir or not os.path.isdir(outdir):
        outdir = os.getcwd()
        
    print("Output folder {}".format(outdir))
      
    # Merge input datasets (Spots are rows and genes are columns)
    counts = aggregate_datatasets(counts_table_files)
    
    # Remove noisy spots and genes (Spots are rows and genes are columns)
    counts = remove_noise(counts, num_exp_genes / 100.0, num_exp_spots / 100.0, 
                          min_expression=min_gene_expression)
    
    # Get the comparisons as tuples
    comparisons = [c.split("-") for c in comparisons]
    
    # Get the conditions 
    conds_repl = dict()
    for cond in conditions:
        d, c = cond.split(":")
        conds_repl[d] = c
    conds = list()
    for spot in counts.index:
        index = spot.split("_")[0]
        try:
            conds.append(conds_repl[index])
        except KeyError:
            counts.drop(spot, axis=0, inplace=True)
            continue

    # Check that the comparisons are valid and if not remove the invalid ones
    comparisons = [c for c in comparisons if c[0] in conds and c[1] in conds]
    if len(comparisons) == 0:
        sys.stderr.write("Error, the vector of comparisons is invalid\n")
        sys.exit(1)
                      
    # Make the DEA call
    print("Doing DEA for the comparisons {} with {} spots and {} genes".format(comparisons,
                                                                              len(counts.index), 
                                                                              len(counts.columns)))   
    # Spots as columns 
    counts = counts.transpose()
    
    # DEA call
    try:
        if normalization in "DESeq2":
            dea_results = deaDESeq2(counts, conds, comparisons, alpha=fdr, size_factors=None)
        else:
            dea_results = deaScranDESeq2(counts, conds, comparisons, alpha=fdr, scran_clusters=False)
    except Exception as e:
        sys.stderr.write("Error while performing DEA " + str(e) + "\n")
        sys.exit(1)
    
    assert(len(comparisons) == len(dea_results))
    for dea_result, comp in zip(dea_results, comparisons):
        # Filter results
        dea_result = dea_result.loc[pd.notnull(dea_result["padj"])]
        dea_result = dea_result.sort_values(by=["padj"], ascending=True, axis=0)
        print("Writing DE genes to output using a FDR cut-off of {}".format(fdr))
        dea_result.to_csv(os.path.join(outdir,
                                       "dea_results_{}_vs_{}.tsv"
                                       .format(comp[0], comp[1])), sep="\t")
        dea_result.ix[dea_result["padj"] <= fdr].to_csv(os.path.join(outdir,
                                                                     "filtered_dea_results_{}_vs_{}.tsv"
                                                                     .format(comp[0], comp[1])), sep="\t")
        # Volcano plot
        print("Writing volcano plot to output")
        outfile = os.path.join(outdir, "volcano_{}_vs_{}.pdf".format(comp[0], comp[1]))
        volcano(dea_result, fdr, outfile)  
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--counts-table-files", required=True, nargs='+', type=str,
                        help="One or more matrices with gene counts per feature/spot (genes as columns)")
    parser.add_argument("--normalization", default="DESeq2", metavar="[STR]", 
                        type=str, 
                        choices=["DESeq2", "Scran"],
                        help="Normalize the counts using:\n" \
                        "DESeq2 = DESeq2::estimateSizeFactors(counts)\n" \
                        "Scran = Deconvolution Sum Factors (Marioni et al)\n" \
                        "(default: %(default)s)")
    parser.add_argument("--conditions", required=True, nargs='+', type=str,
                        help="One of more tuples that represent what conditions to give to each dataset.\n" \
                        "The notation is simple: DATASET:CONDITION DATASET:CONDITION ...\n" \
                        "For example 0:A 1:A 2:B 3:C. Note that datasets numbers start by 0.")
    parser.add_argument("--comparisons", required=True, nargs='+', type=str,
                        help="One of more tuples that represent what comparisons to make in the DEA.\n" \
                        "The notation is simple: CONDITION-CONDITION CONDITION-CONDITION ...\n" \
                        "For example A-B A-C. Note that the conditions must be the same as in the parameter --conditions.")
    parser.add_argument("--num-exp-genes", default=10, metavar="[INT]", type=int, choices=range(0, 100),
                        help="The percentage of number of expressed genes (>= --min-gene-expression) a spot\n" \
                        "must have to be kept from the distribution of all expressed genes (default: %(default)s)")
    parser.add_argument("--num-exp-spots", default=1, metavar="[INT]", type=int, choices=range(0, 100),
                        help="The percentage of number of expressed spots a gene " \
                        "must have to be kept from the total number of spots (default: %(default)s)")
    parser.add_argument("--min-gene-expression", default=1, type=int, choices=range(1, 50),
                        help="The minimum count (number of reads) a gene must have in a spot to be "
                        "considered expressed (default: %(default)s)")
    parser.add_argument("--fdr", type=float, default=0.01,
                        help="The FDR minimum confidence threshold (default: %(default)s)")
    parser.add_argument("--outdir", help="Path to output dir")
    args = parser.parse_args()
    main(args.counts_table_files, args.conditions, args.comparisons, args.outdir,
         args.fdr, args.normalization, args.num_exp_spots, args.num_exp_genes, 
         args.min_gene_expression)