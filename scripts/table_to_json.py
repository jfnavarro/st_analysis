#! /usr/bin/env python
""" 
Script that takes a ST-data file in matrix format (genes as columns and spots as rows)
from the ST Pipeline and converts it a JSON file
The JSON file will be like this :

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
  
The matrix format must be like this:

    gene  gene ...
XxY  count  count
XxY  count  count
... 

@Author Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>
"""

import argparse
import sys
import os
import pandas as pd
import json

def main(data_file, outfile):

    if not os.path.isfile(data_file):
        sys.stderr.write("Error, input file not present or invalid format\n")
        sys.exit(1)
     
    if not outfile:
        outfile = "st_data.json"
    
    # Iterate the table to create JSON records
    counts_table = pd.read_table(data_file, sep="\t", header=0)
    spots = counts_table.indexes
    genes = counts_table.columns
    list_json = list()
    for spot in spots:
        spot.split("x")
        x = spot[0]
        y = spot[1]
        for gene in genes:
            value = counts_table.loc[spot,gene]
            list_json.append({'barcode': "", 'gene': gene, 'x': x, 'y': y, 'hits': value})
  
    # Write JSON records to file
    with open(outfile, "w") as json_handler:
        json.dump(list_json, json_handler, indent=2, separators=(",",": "))  
                     
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file",  required=True,
                        help="ST data file in table format")
    parser.add_argument("--outfile", default=None, help="Name of the output file")
    args = parser.parse_args()
    main(args.data_file, args.outfile)