#! /usr/bin/env python
""" 
Scripts that computes tag clusters counts (TTs)
for ST data. It takes the file generated with compute_st_tts.py
and compute a matrix of counts by doing intersection of the 
TTs with the original ST BED file generate with the ST Pipeline.
The output will look like 

    TT1.....TTN
XxY 
XxY
...

It needs the original BED file with ST data to extract the reads count
If no output file is given the output will be : output_table_ctts.txt

@Author Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>
"""

import argparse
import sys
from collections import defaultdict
import os

def main(input_files, outfile):
    
    if len(input_files) != 2:
        sys.stderr.write("Error, input file not present or invalid format\n")
        sys.exit(1)

    st_bed_file = input_files[1]
    tag_clusters_file = input_files[0]
    
    if not os.path.isfile(st_bed_file) or not os.path.isfile(tag_clusters_file):
        sys.stderr.write("Error, input file not present or invalid format\n")
        sys.exit(1)
     
    if outfile is None:
        outfile = "output_table_ctts.txt"
           
    # load all the original barcode - gene coordinates
    map_original_clusters = defaultdict(list)
    with open(st_bed_file, "r") as filehandler:
        for line in filehandler.readlines():
            if line.find("#") != -1:
                continue
            tokens = line.split()
            assert(len(tokens) == 9)
            chromosome = tokens[0]
            start_site = int(tokens[1])
            end_site = int(tokens[2])
            strand = tokens[5]
            # gene = tokens[6]
            x = float(tokens[7])
            y = float(tokens[8])
            map_original_clusters[(chromosome,strand)].append((x,y,start_site,end_site))
                            
    # loads all the clusters
    map_clusters = defaultdict(int)
    clusters = set()
    barcodes = set()
    with open(tag_clusters_file, "r") as filehandler:
        for line in filehandler.readlines():
            if line.find("#") != -1:
                continue
            tokens = line.split()
            assert(len(tokens) == 8)
            chromosome = tokens[0]
            start = int(tokens[2])
            strand = tokens[1]
            end = int(tokens[3])
            # doing a full search of intersections over all barcodes (similar to bed intersect)
            # If we could rely on that no barcodes were missing doing the clustering we could use
            # a faster approach not needing to iterate all the barcodes but only one   
            # this intersection method is prob overcounting
            for x, y, start_orig, end_orig in map_original_clusters[chromosome, strand]:
                if strand == "-": start_orig = (end_orig - 1)
                if (start_orig >= start and start_orig < end):
                    map_clusters[(x,y,chromosome,strand,start,end)] += 1
                barcodes.add((x,y)) 
            clusters.add((chromosome,strand,start,end))    
    
    # write cluster count for each barcode 
    with open(outfile, "w") as filehandler:
        clusters_string = "\t".join("%s:%s-%s,%s" % cluster for cluster in clusters)
        filehandler.write(clusters_string + "\n")
        for x,y in barcodes:
            filehandler.write("{0}x{1}".format(x,y))
            for chro,strand,star,end in clusters:
                count = map_clusters[(x,y,chro,strand,star,end)]
                filehandler.write("\t{}".format(count))
            filehandler.write("\n")            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_files', nargs=2, 
                        help="The tab delimited file containing the tag clusters and the ST original BED file")
    parser.add_argument("--outfile", default=None, help="Name of the output file")
    args = parser.parse_args()
    main(args.input_files, args.outfile)
