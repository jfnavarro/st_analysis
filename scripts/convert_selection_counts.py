#! /usr/bin/env python
#@Author Jose Fernandez
"""
Script that takes a selection generated
with the ST viewer from a particular dataset 
and the output of the pipeline in BED format
with the new pipeline on the same dataset
and recomputes the gene counts.
"""
import argparse
import sys
import os
from collections import defaultdict

def main(file_selection, bed_file_pipeline, outfile=None):

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
            coordinates.add((int(tokens[1]),int(tokens[2])))

    # loads coordinate-genes from the ST BED file and aggregate counts
    barcodes_genes = defaultdict(int)
    with open(bed_file_pipeline, "r") as filehandler:
        for line in filehandler.readlines():
            if line.find("#") != -1:
                continue
            tokens = line.split()
            gene = str(tokens[6])
            x = int(tokens[7])
            y = int(tokens[8])
            barcodes_genes[(x,y,gene)] += 1
            
    # writes entries that contain a coordiante in the previous list
    with open(outfile, "w") as filehandler_write:
        filehandler_write.write("# gene_name\tx\ty\treads_count\n")
        for x,y,gene,count in barcodes_genes.iteritems():
            if (x,y) in coordinates:
                filehandler_write.write("%s\t%s\t%s\t%s\n" % 
                                        (str(gene), str(x), str(y), str(count)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_files", nargs=2,
                        help="Tab delimited obtained from a selection in the ST "
                        "viewer and the respective ST BED file of that dataset")
    parser.add_argument("--outfile", help="Name of the output file")
    args = parser.parse_args()
    main(args.input_files[0], args.input_files[1], args.outfile)
