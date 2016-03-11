#! /usr/bin/env python
#@Author Jose Fernandez

import argparse
import sys
import os
from collections import defaultdict
from stpipeline.common.utils import fileOk

def main(original_file, barcodes_file, outfile):

    if not fileOk(original_file) or not fileOk(barcodes_file):
        sys.stderr.write("Error, input file not present or invalid format\n")
        sys.exit(-1)
     
    if not outfile:
        outfile = "filtered_" + os.path.basename(original_file)
           
    # loads all the barcodes
    barcodes = dict()
    with open(barcodes_file, "r") as filehandler:
        for line in filehandler.readlines()[1:]:
            tokens = line.split()
            barcodes[tokens[0]] = (tokens[1], tokens[2])
        
    # writes entries that contain a barcode in the previous list
    with open(original_file, "r") as filehandler_read:
        with open(outfile, "w") as filehandler_write:
            for line in filehandler_read.readlines():
                tokens = line.split()
                bc = tokens[1]
                cla = tokens[0]
                x,y = barcodes[bc]
                filehandler_write.write(str(cla) + "\t" + str(bc) + "\t" 
                                        + str(x) + "\t" + str(y) + "\n")
                    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_file", 
                        help="Tab delimited file containing the predicted classes")
    parser.add_argument("--outfile", help="Name of the output file")
    parser.add_argument("--barcodes-file",
                        help="Tab delimited file containing barcodes from the viewer")
    args = parser.parse_args()
    main(args.original_file, args.barcodes_file, args.outfile)


