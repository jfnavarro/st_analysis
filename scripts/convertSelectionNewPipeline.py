#! /usr/bin/env python
#@Author Jose Fernandez
"""
Script that takes a selection make
with the viewer in a dataset with the old pipeline
and the output of the pipeline in BED format
with the new pipeline on the same dataset
and recomputes the selection genes counts.
"""

import argparse
import sys
import os
from collections import defaultdict
from stpipeline.common.utils import fileOk

def main(file_selection, bed_file_pipeline, outfile=None):

    if not fileOk(file_selection) or len(bed_file_pipeline) <= 0:
        sys.stderr.write("Error, input file not present or invalid format\n")
        sys.exit(-1)

    if outfile is None:
        outfile = "adjusted_" + os.path.basename(file_selection)

    # loads all the barcodes from the selection
    barcode_to_cords = dict()
    with open(file_selection, "r") as filehandler:
        for line in filehandler.readlines()[1:]:
            tokens = line.split()
            barcode_to_cords[str(tokens[1])] = (int(tokens[2]),int(tokens[3]))

    # loads barcode-genes from the ST BED file and aggregate counts
    barcodes_genes = defaultdict(int)
    with open(bed_file_pipeline, "r") as filehandler:
        for line in filehandler.readlines()[1:]:
            tokens = line.split()
            barcode = str(tokens[7])
            gene = str(tokens[6])
            barcodes_genes[(barcode,gene)] += 1
            
    # writes entries that contain a barcode in the previous list
    with open(outfile, "w") as filehandler_write:
        filehandler_write.write("# gene_name\tbarcode_id\tx\ty\treads_count\n")
        for key,count in barcodes_genes.iteritems():
            barcode = key[0]
            if barcode in set(barcode_to_cords.keys()):
                x,y = barcode_to_cords[barcode]
                gene = key[1]
                filehandler_write.write(str(gene) + "\t" + str(barcode) + "\t" 
                                        + str(x) + "\t" + str(y) + "\t" + str(count) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_files", nargs=2,
                        help="Tab delimited obtained from a selection in the ST Viewer and the respective ST BED file of that dataset")
    parser.add_argument("--outfile", help="Name of the output file")
    args = parser.parse_args()
    main(args.input_files[0], args.input_files[1], args.outfile)
