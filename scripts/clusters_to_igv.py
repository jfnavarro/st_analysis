#! /usr/bin/env python
#@Author Jose Fernandez
"""
Ugly script that converts the output of countClusters.py
to a format compatible with genome browsers
"""
import sys

if len(sys.argv) == 2:
    input = str(sys.argv[1])
else:
    print "Wrong number of parameters"
    sys.exit(1)

with open(input, "r") as file_read:
    with open("output_clusters_igv.bed", "w") as file_write:
        for line in file_read.readlines()[1:]:
            tokens = line.split()
            file_write.write("chr" + tokens[0] + "\t" + tokens[2] + "\t" + tokens[3] + "\t" \
            + str(tokens[0] + "-" + tokens[2] + "-" + tokens[3] + "." + tokens[1]) \
            + "\t" + tokens[4] + "\t" + tokens[1] + "\n")
