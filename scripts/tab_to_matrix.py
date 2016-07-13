#! /usr/bin/env python
""" 
Script that takes a ST-data file in TAB
format from the ST Viewer and converts it
to a data frame (genes as columns and spots as rows).
The TAB format must be like this :

# gene_name     x       y       reads_count
ENSMUSG00000022756      17      3       3
  ....
  
The output data frame will be like 
    gene  gene ...
XxY  count  count
XxY  count  count
... 

@Author Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>
"""

import argparse
import sys
import os
from collections import defaultdict
import pandas as pd

def main(tab_file, outfile):

    if not os.path.isfile(tab_file):
        sys.stderr.write("Error, input file not present or invalid format\n")
        sys.exit(1)
     
    if not outfile:
        outfile = "data_table.tsv"
    
    # Iterate the JSON file to get the counts   
    genes_spot_counts = defaultdict(lambda : defaultdict(int))
    with open(tab_file, "r") as fh:
        for line in fh.readlines()[1:]:
            tokens = line.split()
            gene = tokens[0]
            x = tokens[1]
            y = tokens[2]
            count = tokens[3]
            spot = "{0}x{1}".format(x, y)
            genes_spot_counts[spot][gene] = count
    
    # Obtain a list of the row names (indexes) 
    # and list of list of gene->count for the columns (of each row)
    list_row_values = list()
    list_indexes = list()    
    for key,value in genes_spot_counts.iteritems():
        list_indexes.append(key)
        list_row_values.append(value)
        
    # Create a data frame (genes as columns, spots as rows)
    counts_table = pd.DataFrame(list_row_values, index=list_indexes)
    # Write table to a file
    counts_table.to_csv(outfile, sep="\t", na_rep=0)
               
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tab-file",  required=True,
                        help="ST data file in TAB format")
    parser.add_argument("--outfile", default=None, help="Name of the output file")
    args = parser.parse_args()
    main(args.tab_file, args.outfile)