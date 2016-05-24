#! /usr/bin/env python
#@Author Jose Fernandez
""" 
Script that takes a ST-data file in JSON
format from the ST Pipeline and converts it
to a data frame
"""

import argparse
import sys
import os
from collections import defaultdict
import pandas as pd
import json

def main(json_file, outfile):

    if not os.path.isfile(json_file) or not json_file.tolower().endswith(".json"):
        sys.stderr.write("Error, input file not present or invalid format\n")
        sys.exit(1)
     
    if not outfile:
        outfile = "data_table.csv"
    
    # Iterate the JSON file to get the counts   
    genes = set()
    spots = set()
    genes_spot_counts = defaultdict(int)
    with open(json_file, "r") as fh:
        for line in json.load(fh):
            gene = line["gene"]
            x = int(line["x"])
            y = int(line["y"])
            count = line["hits"]
            genes.add(gene)
            spots.add((x,y))
            genes_spot_counts[(x,y),gene] = count
    
    # Create a data frame with the counts (genes as columns, spots as rows)
    counts_table = pd.DataFrame(genes_spot_counts)
    # Write table to a ile
    counts_table.to_csv(outfile)
               
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-file",  required=True,
                        help="ST data file in JSON format")
    parser.add_argument("--outfile", default=None, help="Name of the output file")
    args = parser.parse_args()
    main(args.json_file, args.outfile)