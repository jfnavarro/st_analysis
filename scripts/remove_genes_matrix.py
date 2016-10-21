#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
A script that removes genes from the matrix
given as input that match the regular expression given as input.

@Author Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>
"""
import argparse
import sys
import os
import pandas as pd
import re

def main(counts_matrix, 
         reg_exp,
         outfile):

    if not os.path.isfile(counts_matrix):
        sys.stderr.write("Error, input file/s not present or invalid format\n")
        sys.exit(1)
            
    if outfile is None: 
        outfile = "adjusted_{}".format(os.path.basename(counts_matrix))
     
    counts = pd.read_table(counts_matrix, sep="\t", header=0, index_col=0)
    genes = counts.columns
    drop_genes = [gene for gene in genes if re.match(reg_exp, gene)]  
    counts.drop(drop_genes, axis=1, inplace=True)
    counts.to_csv(outfile, sep="\t")
                         
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--counts-matrix", required=True, type=str,
                        help="A matrix with counts (Genes as columns and spots as rows)")
    parser.add_argument("--reg-exp", default="", metavar="[STR]", type=str,
                        help="A regular expression to be used to remove the genes that match")
    parser.add_argument("--outfile", default=None, help="Name for the output file")
    args = parser.parse_args()
    main(args.counts_matrix, args.reg_exp, args.outfile)

#
