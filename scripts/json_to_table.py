#! /usr/bin/env python
#@Author Jose Fernandez
""" 
Script that takes a ST-data file in JSON
format from the ST Pipeline and converts it
to a data frame (genes as columns and spots as rows).
The JSON format must be like this :

[
  {
    "y": 25,
    "x": 31,
    "hits": 1,
    "barcode": "GATCGCTGAAAGGATAGA",
    "gene": "ENSMUSG00000041378"
  },
  {
    "y": 23,
    "x": 13,
    "hits": 4,
    "barcode": "TGTTCCGATGGGAGAAGC",
    "gene": "ENSMUSG00000001227"
  },
  ....
  
@Author Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>
"""

import argparse
import sys
import os
from collections import defaultdict
import pandas as pd
import json

def main(json_file, outfile):

    if not os.path.isfile(json_file) or not json_file.endswith(".json"):
        sys.stderr.write("Error, input file not present or invalid format\n")
        sys.exit(1)
     
    if not outfile:
        outfile = "data_table.tsv"
    
    # Iterate the JSON file to get the counts   
    genes_spot_counts = defaultdict(lambda : defaultdict(int))
    with open(json_file, "r") as fh:
        for line in json.load(fh):
            gene = line["gene"]
            x = line["x"]
            y = line["y"]
            count = line["hits"]
            spot = "%sx%s" % (x, y)
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
    print counts_table.index
    # Write table to a ile
    counts_table.to_csv(outfile, sep="\t", na_rep=0)
               
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-file",  required=True,
                        help="ST data file in JSON format")
    parser.add_argument("--outfile", default=None, help="Name of the output file")
    args = parser.parse_args()
    main(args.json_file, args.outfile)