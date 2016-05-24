#! /usr/bin/env python
"""
Script that takes a selection generated
with the ST viewer from a particular dataset 
and the output of the ST pipeline in BED format
from the same dataset and recomputes the gene counts
to generate a new ST viewer selection file.
Reason to do this is if the ST Pipeline was
run again with newer version or different parameters
on the same dataset.

Selection must have the following format :

GENE X Y COUNT

BED file must have the following format

CHROMOSOME START END READ SCORE STRAND GENE X Y

@Author Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>
"""
import argparse
import sys
import os
from collections import defaultdict

def main(file_selection, bed_file_pipeline, outfile):

    if not os.path.isfile(file_selection) or not os.path.isfile(bed_file_pipeline):
        sys.stderr.write("Error, input file not present or invalid format\n")
        sys.exit(1)

    if outfile is None:
        outfile = "adjusted_" + os.path.basename(file_selection)

    # loads all the coordinates from the selection
    coordinates = set()
    with open(file_selection, "r") as filehandler:
        for line in filehandler.readlines():
            if line.find("#") != -1:
                continue
            tokens = line.split()
            assert(len(tokens) == 4)
            coordinates.add((int(tokens[1]),int(tokens[2])))

    # loads coordinate-genes from the ST BED file and aggregate counts
    barcodes_genes = defaultdict(int)
    with open(bed_file_pipeline, "r") as filehandler:
        for line in filehandler.readlines():
            if line.find("#") != -1:
                continue
            assert(len(tokens) == 9)
            tokens = line.split()
            gene = str(tokens[6])
            x = int(tokens[7])
            y = int(tokens[8])
            barcodes_genes[(x,y,gene)] += 1
            
    # writes entries that contain a coordiante in the previous list
    with open(outfile, "w") as filehandler_write:
        filehandler_write.write("# gene_name\tx\ty\treads_count\n")
        for (x,y,gene),count in barcodes_genes.iteritems():
            if (x,y) in coordinates:
                filehandler_write.write("%s\t%s\t%s\t%s\n" % 
                                        (str(gene), str(x), str(y), str(count)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer-selection", required=True,
                        help="A selection made in the ST Viewer")
    parser.add_argument("--bed-file", required=True,
                        help="A bed file from the ST Pipeline (x,y) coordinates must be in the file")
    parser.add_argument("--outfile", default=None, help="Name of the output file")
    args = parser.parse_args()
    main(args.viewer_selection, args.bed_file, args.outfile)
