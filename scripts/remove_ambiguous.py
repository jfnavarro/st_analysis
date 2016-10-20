#! /usr/bin/env python
# -*- coding: utf-8 -*-
""" 
Script that takes a ST-data file in JSON
format from the ST Pipeline and generates
a new JSON file with no ambiguous genes

The JSON file must be like :

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
import json

def main(json_file, outfile):

    if not os.path.isfile(json_file) or not json_file.endswith(".json"):
        sys.stderr.write("Error, input file not present or invalid format\n")
        sys.exit(1)
     
    if not outfile:
        outfile = "st_data_adjusted.json"
            
    # Iterate the JSON file to remove ambiguous genes
    adjusted_list = list()
    with open(json_file, "r") as fh:
        for line in json.load(fh):
            if line["gene"].find("__ambiguous") == -1 and int(line["hits"]) != 0:
                line["x"] = float(line["x"])
                line["y"] = float(line["y"])
                adjusted_list.append(line)
    
    # Write well formed json file
    with open(outfile, "w") as filehandler:
        json.dump(adjusted_list, filehandler, indent=2, separators=(',', ': '))  
               
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json-file",  required=True,
                        help="ST data file in JSON format")
    parser.add_argument("--outfile", default=None, help="Name of the output file")
    args = parser.parse_args()
    main(args.json_file, args.outfile)
