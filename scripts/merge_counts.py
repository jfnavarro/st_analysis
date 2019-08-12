#! /usr/bin/env python
""" 
This script simply merges ST matrices of counts into one 
(adding an index to the spots of each matrix)

@Author Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>
"""

import argparse
import sys
import os
import pandas as pd
from stanalysis.preprocessing import aggregate_datatasets

def main(counts_files, outfile):

    if len(counts_files) == 0 or \
    any([not os.path.isfile(f) for f in counts_files]):
        sys.stderr.write("Error, input file/s not present or invalid format\n")
        sys.exit(1) 
     
    if not outfile:
        outfile = "merged_counts.tsv"
    print("Input datasets {}".format(" ".join(counts_files))) 
    
    # Merge input datasets (Spots are rows and genes are columns)
    counts = aggregate_datatasets(counts_files)
    print("Total number of spots {}".format(len(counts.index)))
    print("Total number of genes {}".format(len(counts.columns)))
    
    # Write filtered table
    counts.to_csv(outfile, sep='\t')
               
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--counts-files", required=True, nargs='+', type=str,
                        help="One or more matrices with gene counts per spot (genes as columns)")
    parser.add_argument("--outfile", help="Name of the output file")
    args = parser.parse_args()
    main(args.counts_files, args.outfile)


