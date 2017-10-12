#! /usr/bin/env python
"""
This scripts merges two ST datasets (technical replicates
from the same individual).

It keeps only the genes that are in both datasets
(summing their counts or averaging them).

Assumes that both matrices have the same order of genes and spots
and that the spots of both datasets are located in the same part of the tissue (aligned).

The spots coordinates of the merged dataset will be the ones present in the first
dataset.

merge_replicates.py --input-files datasetA.tsv datasetB.tsv --output merged.tsv

@Author Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>
"""

import argparse
import sys
import os
import pandas as pd
from stanalysis.preprocessing import merge_datasets

def main(input_files, outfile, merging_action):

    if len(input_files) != 2 or any([not os.path.isfile(f) for f in input_files]):
        sys.stderr.write("Error, input file not present or invalid format\n")
        sys.exit(1)
     
    if not outfile:
        outfile = "merged.tsv"
    
    # Read the data frames (genes as columns)
    counts_tableA = pd.read_table(input_files[0], sep="\t", header=0, index_col=0)
    counts_tableB = pd.read_table(input_files[1], sep="\t", header=0, index_col=0)

    num_spotsA = len(counts_tableA.index)
    num_spotsB = len(counts_tableB.index)
    if num_spotsA != num_spotsB:
        sys.stderr.write("Error, datasets have different number of spots "
                         "{} and {}\n".format(num_spotsA, num_spotsB))
        sys.exit(1)
         
    print("Merging dataset {} with {} spots and {} genes with "
          "dataset {} with {} spots and {} genes".format(input_files[0], input_files[1], 
                                                         num_spotsA, num_spotsB))
    # Merge the two datasets
    merged_table = merge_datasets(counts_tableA, counts_tableB, merging_action)
    
    # Write merged table
    merged_table.to_csv(outfile, sep='\t')
               
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-files", required=True, nargs='+', type=str,
                        help="Two ST datasets (matrix of counts in TSV format)")
    parser.add_argument("--outfile", help="Name of the output file")
    parser.add_argument("--merging-action", default="Sum", metavar="[STR]", 
                        type=str, choices=["Sum", "Median"],
                        help="How to merge the counts of common genes in both datasets.\n"
                        "Sum will sum the counts of both and Median will sum the counts and "
                        "divided by 2 (default: %(default)s).")
    args = parser.parse_args()
    main(args.input_files, args.outfile, args.merging_action)

