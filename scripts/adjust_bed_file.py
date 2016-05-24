#! /usr/bin/env python
#@Author Jose Fernandez
""" 
Script that takes a BED file from older
versions of the ST pipeline and replaces
the BARCODE field for the x,y coordinates

Old BED file :

CHROMOSOME START END READ SCORE STRAND GENE BARCODE

New BED file must have the following format

CHROMOSOME START END READ SCORE STRAND GENE X Y

It requires a file with the BARCODE ids and the coordinates as 

BARCODE X Y

"""

import argparse
import sys
import os

def main(bed_file, barcode_ids, outfile):

    if not os.path.isfile(bed_file) or not os.path.isfile(barcode_ids):
        sys.stderr.write("Error, input file not present or invalid format\n")
        sys.exit(1)
     
    if not outfile:
        outfile = "adjusted_" + os.path.basename(bed_file)
           
    # loads all the barcodes
    barcodes = dict()
    with open(barcode_ids, "r") as filehandler:
        for line in filehandler.readlines():
            tokens = line.split()
            barcodes[tokens[0]] = (tokens[1],tokens[2])
    
    # Read the bed file and write out the adjusted one
    with open(bed_file, "r") as filehandler_read:
        with open(outfile, "w") as filehandler_write:
            for line in filehandler_read.readlines()[1:]:
                tokens = line.split()
                bc = tokens[7]
                x,y = barcodes[bc]
                filehandler_write.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % 
                                        (tokens[0], tokens[1], tokens[2], tokens[3], 
                                         tokens[4], tokens[5], tokens[6], str(x), str(y)))
               
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bed-file", required=True,
                        help="BED file from the ST Pipeline (older versions with barcode tag instead of x and y)")
    parser.add_argument("--outfile", help="Name of the output file")
    parser.add_argument("--barcodes-ids", required=True,
                        help="File with the barcode ids and their coordinates")
    args = parser.parse_args()
    main(args.bed_file, args.barcodes_ids, args.outfile)


