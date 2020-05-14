#! /usr/bin/env python
""" 
This script simply merges matrices of counts (ST datasets) into a combined one.\n
The merging is done by rows adding an optional index to the rows that corresponds to the order in the input.

@Author Jose Fernandez Navarro <jc.fernandez.navarro@gmail.com>
"""

import argparse
import sys
import os
import pandas as pd
from stanalysis.preprocessing import aggregate_datatasets

def main(counts_files, no_header, no_index, outfile):

    if len(counts_files) == 0 or \
    any([not os.path.isfile(f) for f in counts_files]):
        sys.stderr.write("Error, input file/s not present or invalid format\n")
        sys.exit(1) 
     
    if not outfile:
        outfile = "merged_counts.tsv"
    print("Input datasets {}".format(" ".join(counts_files))) 
    
    # Merge input datasets by rows
    counts = aggregate_datatasets(counts_files, 
                                  add_index=not no_index, 
                                  header=None if no_header else 0)
    print("Total number of spots {}".format(len(counts.index)))
    print("Total number of genes {}".format(len(counts.columns)))
    
    # Write filtered table
    counts.to_csv(outfile, sep='\t', header=not no_header)
               
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--counts-files", required=True, nargs='+', type=str,
                        help="One or more matrices of counts (spots as rows and genes as columns)")
    parser.add_argument("--no-header", action="store_true", default=False,
                        help="Use this flag if the input matrices do not contain a header")
    parser.add_argument("--no-index", action="store_true", default=False,
                        help="Use this flag to not add an index to the rows in the merged matrix (one index per input dataset)")
    parser.add_argument("--outfile", help="Name of the output file")
    args = parser.parse_args()
    main(args.counts_files, args.no_header, args.no_index, args.outfile)


