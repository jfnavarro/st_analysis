#! /usr/bin/env python
""" 
Script that takes a ST-data file in JSON
format from the ST Pipeline and generates
a new JSON file with the pixel image coordinates instead
of array coordinates. For this a tab delimited
file with the following format is needed:

array_x array_y pixel_x pixel_y

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
from collections import defaultdict
import json

def main(json_file, coordinates_file, outfile):

    if not os.path.isfile(json_file) or not json_file.endswith(".json") \
    or not os.path.isfile(coordinates_file):
        sys.stderr.write("Error, input file not present or invalid format\n")
        sys.exit(1)
     
    if not outfile:
        outfile = "st_data_adjusted.json"
    
    # Get a map of the new coordinates
    new_coordinates = dict()
    with open(coordinates_file, "r") as filehandler:
        for line in filehandler.readlines():
            tokens = line.split()
            assert(len(tokens) == 4)
            old_x = int(tokens[0])
            old_y = int(tokens[1])
            new_x = float(tokens[2])
            new_y = float(tokens[3])
            new_coordinates[(old_x, old_y)] = (new_x,new_y)
            
    # Iterate the JSON file to update the X,Y coordinates
    adjusted_list = list()
    with open(json_file, "r") as fh:
        for line in json.load(fh):
            old_x = int(line["x"])
            old_y = int(line["y"])
            if (old_x,old_y) in new_coordinates:
                new_x, new_y = new_coordinates[(old_x,old_y)]
                line["x"] = new_x
                line["y"] = new_y
                adjusted_list.append(line)
    
    # Write well formed json file
    with open(outfile, "w") as filehandler:
        json.dump(adjusted_list, filehandler, indent=2, separators=(',', ': '))  
               
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-file",  required=True,
                        help="ST data file in JSON format")
    parser.add_argument("--coordinates-file",  required=True,
                        help="New coordinates in a tab delimited file")
    parser.add_argument("--outfile", default=None, help="Name of the output file")
    args = parser.parse_args()
    main(args.json_file, args.coordinates_file, args.outfile)