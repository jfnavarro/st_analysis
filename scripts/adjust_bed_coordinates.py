#! /usr/bin/env python
""" 
Script that takes the ST BED file generated
from the ST pipeline and a tab delimited
file with new spot coordinates and
replace the coordinates in the BED files
for the new ones.
The tab delimited file must have the following format : 

old_x old_y new_x new_y

The format of the BED file :

CHROMOSOME START END READ SCORE STRAND GENE X Y


@Author Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>
"""

import argparse
import sys
import os

def main(bed_file, coordinates_file, outfile):

    if not os.path.isfile(bed_file) or not os.path.isfile(coordinates_file):
        sys.stderr.write("Error, input file not present or invalid format\n")
        sys.exit(1)
     
    if not outfile:
        outfile = "adjusted_{}".format(os.path.basename(bed_file))
           
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
        
    # Writes entries with new coordiantes
    with open(bed_file, "r") as filehandler_read:
        with open(outfile, "w") as filehandler_write:
            for line in filehandler_read.readlines():
                if line.find("#") != -1:
                    continue
                tokens = line.split()
                assert(len(tokens) == 9)
                x = int(tokens[7])
                y = int(tokens[8])
                if (x,y) in new_coordinates:
                    tokens[7], tokens[8] = new_coordinates[(x,y)]
                    filehandler_write.write("\t".join([str(ele) for ele in tokens]) + "\n")
                    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bed_file", 
                        help="Tab delimited file containing the ST data in BED format")
    parser.add_argument("--outfile", help="Name of the output file")
    parser.add_argument("--coordinates-file",  required=True,
                        help="New coordinates in a tab delimited file")
    args = parser.parse_args()
    main(args.bed_file, args.coordinates_file, args.outfile)

