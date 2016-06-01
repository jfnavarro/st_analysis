#! /usr/bin/env python
""" 
Script that takes the ST BED file generated
from the ST pipeline and a one or more selections
made in the ST Viewer and generate a new file
with only the coordinates that are present in the selections.

The format of the BED file :

CHROMOSOME START END READ SCORE STRAND GENE X Y

The format of ST viewer selection

GENE X Y COUNT

@Author Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>
"""

import argparse
import sys
import os

def main(bed_file, barcodes_files, outfile):

    if not os.path.isfile(bed_file) or len(barcodes_files) <= 0:
        sys.stderr.write("Error, input file not present or invalid format\n")
        sys.exit(1)
     
    if not outfile:
        outfile = "filtered_{}".format(os.path.basename(bed_file))
           
    # loads all the coordinates from the ST Viewer selection
    barcodes = set()
    for barcode_file in barcodes_files:
        with open(barcode_file, "r") as filehandler:
            for line in filehandler.readlines():
                if line.find("#") == -1:
                    tokens = line.split()
                    assert(len(tokens) == 4)
                    x = int(tokens[1])
                    y = int(tokens[2])
                    barcodes.add((x,y))
        
    # Writes entries that contain a coordinate in the previous list
    with open(bed_file, "r") as filehandler_read:
        with open(outfile, "w") as filehandler_write:
            for line in filehandler_read.readlines():
                if line.find("#") != -1:
                    continue
                tokens = line.split()
                assert(len(tokens) == 9)
                x = int(tokens[7])
                y = int(tokens[8])
                if (x,y) in barcodes:
                    filehandler_write.write(line)
                    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bed_file", 
                        help="Tab delimited file containing the ST data in BED format")
    parser.add_argument("--outfile", help="Name of the output file")
    parser.add_argument("--barcodes-files", nargs='+', type=str, required=True,
                        help="Tab delimited file containing a selection from the ST viewer")
    args = parser.parse_args()
    main(args.bed_file, args.barcodes_files, args.outfile)

