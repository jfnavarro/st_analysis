#! /usr/bin/env python
""" 
Script that takes a matrix of counts
where the columns are genes and the rows
are BARCODE ids like
        gene    gene    
BARCODE
BARCODE

and convert the ids to XxY coordinates like 
    gene    gene
XxY
XxY

For that it needs a file with the barcode ids and the coordinates as

BARCODE X Y

@Author Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>
"""

import argparse
import sys
import os
import pandas as pd

def main(counts_matrix, barcode_ids, outfile):

    if not os.path.isfile(counts_matrix) or not os.path.isfile(barcode_ids):
        sys.stderr.write("Error, input file not present or invalid format\n")
        sys.exit(1)
     
    if not outfile:
        outfile = "filtered_{}".format(os.path.basename(counts_matrix))
           
    # loads all the barcodes
    barcodes = dict()
    with open(barcode_ids, "r") as filehandler:
        for line in filehandler.readlines():
            tokens = line.split()
            assert(len(tokens) == 3)
            barcodes[tokens[0]] = (tokens[1],tokens[2])
    
    # Read the data frame (barcodes as rows
    counts_table = pd.read_table(counts_matrix, sep="\t", header=0, index_col=0)
    new_index_values = list()
    # Replace barcode for coordinates
    for bc in counts_table.index:
        (x,y) = barcodes[bc]
        new_index_values.append("{0}x{1}".format(x,y))
    # Write table again
    counts_table.index = new_index_values
    counts_table.to_csv(outfile, sep='\t')
               
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts-matrix", required=True,
                        help="Matrix with gene counts (genes as columns)")
    parser.add_argument("--outfile", help="Name of the output file")
    parser.add_argument("--barcodes-ids", required=True,
                        help="File with the barcode ids and their coordinates")
    args = parser.parse_args()
    main(args.counts_matrix, args.barcodes_ids, args.outfile)

