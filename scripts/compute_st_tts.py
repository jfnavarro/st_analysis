#! /usr/bin/env python
""" 
Script that takes as input a BED file generated
from the ST Pipeline and computes peaks clusters
based on the transcription termination site and the strand. 
It uses paraclu to compute the clusters
and paraclu-cut to filter out (optional)
The computed ST-TTs are written to a file in BED format. 

The ST BED file must has the following format:

CHROMOSOME START END READ SCORE STRAND GENE X Y

@Author Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>
"""

import argparse
import sys
import os
from collections import defaultdict
import subprocess
import tempfile

def paracluFilter(file_input, file_output, max_cluster_size, 
                  min_density_increase, single_cluster_density):
    """ A simple wrapper to run paraclu-cut previous sorting of the file
    """
    # sort before filter
    temp_filtered = tempfile.mktemp(prefix="st_countClusters_filtered")
    with open(temp_filtered, "w") as filehandler:
        args = ['sort']
        args += ["-k1,1"]
        args += ["-k2,2"]
        args += ["-k3n,3"]
        args += ["-k4nr,4"]
        args += [file_input]
        try:
            proc = subprocess.Popen([str(i) for i in args], 
                                    stdout=filehandler, stderr=subprocess.PIPE)
            (stdout, errmsg) = proc.communicate()
        except Exception:
            raise
    
    if not os.path.isfile(temp_filtered):
        raise RuntimeError("Error sorting..\n{0}\n{1}\n".format(stdout, errmsg))
                    
    # call paraclu-cut to filter out the clusters
    with open(file_output, "w") as filehandler:
        args = ['paraclu-cut.sh']
        args += ["-l", max_cluster_size]
        args += ["-d", min_density_increase]
        args += [temp_filtered]
        try:
            proc = subprocess.Popen([str(i) for i in args], 
                                    stdout=filehandler, stderr=subprocess.PIPE)
            (stdout, errmsg) = proc.communicate()
        except Exception:
            raise
    
    if not os.path.isfile(file_output):
        raise RuntimeError("Error running paraclu-cut..\n{0}\n{1}\n".format(stdout, errmsg))
                         
def main(bed_file, min_data_value, disable_filter, 
         max_cluster_size, min_density_increase, output):

    if not os.path.isfile(bed_file):
        sys.stderr.write("Error, input file not present or invalid format\n")
        sys.exit(1)
     
    if output is None: 
        output = "filtered_tag_clusters.bed"
         
    # First we parse the BED file to group by chromsome,strand and star_site
    # Input file has the following format (this is the first line)
    # Chromosome Start End Read Score Strand Gene Barcode
    print "Grouping entries by Chromosome, strand and start site..."
    map_reads = defaultdict(int)
    with open(bed_file, "r") as file_handler:
        for line in file_handler.readlines():
            if line.find("#") != -1:
                continue
            tokens = line.split()
            assert(len(tokens) == 9)
            chromosome = tokens[0]
            star_site = int(tokens[1])
            end_site = int(tokens[2])
            strand = tokens[5]
            #gene = tokens[6]
            # Swap star and end site if the gene is annotated in the negative strand
            if strand == "-": star_site = end_site
            map_reads[(chromosome,strand,star_site)] += 1
     
    print "Writing grouped entries to a temp file..."   
    temp_grouped_reads = tempfile.mktemp(prefix="st_countClusters_grouped_reads")
    with open(temp_grouped_reads, "w") as filehandler:
        # iterate the maps to write out the grouped entries with the barcodes
        for key,value in map_reads.iteritems():
            chromosome = key[0]
            strand = key[1]
            star_site = key[2]
            filehandler.write("{0}\t{1}\t{2}\t{3}\n".format(chromosome, strand, star_site, value))

    if not os.path.isfile(temp_grouped_reads):
        sys.stderr.write("Error, no entries found in input file {}".format(bed_file))
        sys.exit(1)
        
    # Sort the grouped entries
    temp_grouped_reads_sorted = tempfile.mktemp(prefix="st_countClusters_grouped_sorted_reads")
    print "Sorting the grouped entries to {}".format(temp_grouped_reads_sorted)
    with open(temp_grouped_reads_sorted, "w") as filehandler:
        args = ['sort']
        args += ["-k1,1"]
        args += ["-k2,2"]
        args += ["-k3n,3"]
        args += [temp_grouped_reads]
        try:
            proc = subprocess.Popen([str(i) for i in args], 
                                    stdout=filehandler, stderr=subprocess.PIPE)
            (stdout, errmsg) = proc.communicate()
        except Exception as e:
            sys.stderr.write("Error sorting\n")
            sys.stderr.write(str(e))
            sys.exit(1)    
    
    if not os.path.isfile(temp_grouped_reads_sorted):
        sys.stderr.write("Error sorting \n{}\n{}\n".format(stdout, errmsg))
        sys.exit(1)
        
    # call paraclu to compute peak clusters  
    temp_paraclu = tempfile.mktemp(prefix="st_countClusters_paraclu")
    print "Making the peaks calling with paraclu to {}".format(temp_paraclu)
    with open(temp_paraclu, "w") as filehandler:
        args = ['paraclu']
        args += [min_data_value]
        args += [temp_grouped_reads_sorted]
        try:
            proc = subprocess.Popen([str(i) for i in args], 
                                    stdout=filehandler, stderr=subprocess.PIPE)
            (stdout, errmsg) = proc.communicate()
        except Exception as e:
            sys.stderr.write("Error executing paraclu\n")
            sys.stderr.write(str(e))
            sys.exit(1)
      
    if not os.path.isfile(temp_paraclu):
        sys.stderr.write("Error executing paraclu \n{}\n{}\n".format(stdout, errmsg))
        sys.exit(1)
              
    if not disable_filter:
        print "Filtering computed clusters with paraclu-cut..."
        try:
            paracluFilter(temp_paraclu, output, 
                          max_cluster_size, min_density_increase, False)
        except Exception as e:
            sys.stderr.write("Error, executing paraclu-cut\n")
            sys.stderr.write(str(e))
            sys.exit(1)
    else:
        os.rename(temp_paraclu, output)
        
    print "DONE!"
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bed_file", help="BED ST-data file")
    parser.add_argument("--min-data-value", default=20, metavar="[INT]", type=int, choices=range(1, 999),
                        help="Omits grouped entries whose total count is lower than this (default: %(default)s)")
    parser.add_argument("--disable-filter", action="store_true", 
                        default=False, help="Disable second filter(paraclu-cut)")
    parser.add_argument("--max-cluster-size", default=200, metavar="[INT]", type=int, choices=range(10, 999),
                        help="Discard clusters whose size in positions is bigger than this (default: %(default)s)")
    parser.add_argument("--min-density-increase", default=2, metavar="[INT]", type=int, choices=range(1, 99),
                        help="Discard clusters whose density is lower than this")
    parser.add_argument("--output", default=None, help="The name and path of the output file (default: %(default)s)")
    args = parser.parse_args()
    main(args.bed_file, args.min_data_value, args.disable_filter, 
         args.max_cluster_size, args.min_density_increase, args.output)

