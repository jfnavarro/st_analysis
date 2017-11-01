#! /usr/bin/env python
""" 
Script that takes a ST dataset (matrix of counts)
where the columns are genes and the rows
are spot coordinates
        gene    gene    
XxY
XxY
...

and a file of spot classes

XxY 1
XxY 1
XxY 2
...

And slices the matrix into regions given as input

1 2 ...

slice_regions_matrix.py --counts-matrix dataset.tsv --spot-classes classes.txt --regions 1 3

@Author Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>
"""

import argparse
import sys
import os
import pandas as pd
from collections import defaultdict

def main(counts_matrix, class_file, regions):

    if not os.path.isfile(counts_matrix) \
    or not os.path.isfile(class_file) or len(regions) == 0:
        sys.stderr.write("Error, input file not present or invalid format\n")
        sys.exit(1)
    
    # Get the file name
    base_name = os.path.basename(counts_matrix).split(".")[0]
    # Read the data frame (genes as columns)
    counts_table = pd.read_table(counts_matrix, sep="\t", header=0, index_col=0)
    # Load the spot classes
    spot_classes = defaultdict(list)
    with open(class_file) as filehandler:
        for line in filehandler.readlines():
            tokens = line.split()
            assert(len(tokens) == 2)
            # Assure  spots have two decimals
            x = round(float(tokens[0].split("x")[0]), 2)
            y = round(float(tokens[0].split("x")[1]), 2)
            spot = "{}x{}".format(x,y)
            spot_classes[tokens[1]].append(spot)
    # Iterate the regions and slice the matrix
    for region, spots in spot_classes.items():
        if region in regions:
            slice = counts_table.loc[spots]
            slice.to_csv("{}_{}.tsv".format(base_name, region), sep='\t')
               
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--counts-matrix", required=True,
                        help="Matrix with gene counts (genes as columns)")
    parser.add_argument("--spot-classes", type=str, 
                        help="Path to the file containing the spot classes as\nSPOT INTEGER")
    parser.add_argument("--regions", 
                        help="The regions (CLASSES) to split the dataset into",
                        required=True, nargs='+', type=str)
    args = parser.parse_args()
    main(args.counts_matrix, args.spot_classes, args.regions)

