#! /usr/bin/env python
#@Author Jose Fernandez
""" 
Script that takes a matrix of counts
where the columns are barcode Ids and convert
the ids to XxY coordinates. For that it also needs
a file with the barcode ids and the coordinates
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
        outfile = "filtered_" + os.path.basename(counts_matrix)
           
    # loads all the barcodes
    barcodes = dict()
    with open(barcode_ids, "r") as filehandler:
        for line in filehandler.readlines():
            tokens = line.split()
            barcodes[tokens[0]] = (tokens[1],tokens[2])
    
    # Read the data frame
    counts_table = pd.read_table(counts_matrix, sep="\t", header=0)
    transpose_counts_table = counts_table.transpose()
    column_values = list(transpose_counts_table.columns.values)
    new_column_values = list()
    # Replace barcode for coordinates
    for bc in column_values:
        (x,y) = barcodes[bc]
        new_column_values.append(x + "x" + y)
    # Write table again
    transpose_counts_table.columns = new_column_values
    transpose_counts_table.to_csv(outfile)
               
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts-matrix",  
                        help="Matrix with gene counts (genes as columns)")
    parser.add_argument("--outfile", help="Name of the output file")
    parser.add_argument("--barcodes-ids",
                        help="File with the barcode ids and their coordinates")
    args = parser.parse_args()
    main(args.counts_matrix, args.barcodes_ids, args.outfile)

