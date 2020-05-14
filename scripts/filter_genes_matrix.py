#! /usr/bin/env python
""" 
Script that takes a ST dataset (matrix of counts)
where the columns are genes and the rows
are spot coordinates
        gene    gene    
XxY
XxY

And removes the columns of genes
matching the regular expression given as input.

@Author Jose Fernandez Navarro <jc.fernandez.navarro@gmail.com>
"""

import argparse
import sys
import os
import pandas as pd
import re

def main(counts_matrix, reg_exps, outfile):

    if not os.path.isfile(counts_matrix):
        sys.stderr.write("Error, input file not present or invalid format\n")
        sys.exit(1)
     
    if not outfile:
        outfile = "filtered_{}.tsv".format(os.path.basename(counts_matrix).split(".")[0])
    
    # Read the data frame (genes as columns)
    counts_table = pd.read_csv(counts_matrix, sep="\t", header=0, index_col=0)
    genes = counts_table.columns
    # Filter out genes that match any of the reg-exps
    genes = [gene for gene in genes if any([re.match(regex,gene) for regex in reg_exps])]
    counts_table.drop(genes, axis=1, inplace=True)
    # Write filtered table
    counts_table.to_csv(outfile, sep='\t')
               
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--counts-matrix", required=True,
                        help="Matrix with gene counts (genes as columns)")
    parser.add_argument("--outfile", help="Name of the output file")
    parser.add_argument("--filter-genes", help="Regular expression for \
                        gene symbols to filter out. Can be given several times.",
                        default=None,
                        type=str,
                        action='append')
    args = parser.parse_args()
    main(args.counts_matrix, args.filter_genes, args.outfile)

