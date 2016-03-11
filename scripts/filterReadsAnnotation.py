#! /usr/bin/env python
#@Author Jose Fernandez
""" 
Script that takes the reads.bed file generated
from the ST pipeline and a the annotation
file used in the ST pipeline to apply filters
to the reads.bed file based of type of transcript
"""

import argparse
import sys
import os
from collections import defaultdict
from stpipeline.common.utils import fileOk
filter = ['polymorphic_pseudogene', 'processed_transcript', 'protein_coding', 'lincRNA', "3prime_overlapping_ncrna", "miRNA"]
#set(['unitary_pseudogene', 'rRNA', 'IG_D_pseudogene', 'lincRNA', 'IG_C_pseudogene', 'translated_processed_pseudogene', 
#'Mt_tRNA', 'sense_intronic', 'IG_V_gene', 'misc_RNA', 'polymorphic_pseudogene', 'IG_J_gene', 'TR_J_pseudogene', 
#'IG_LV_gene', 'TEC', 'protein_coding', 'Mt_rRNA', 'TR_V_pseudogene', '3prime_overlapping_ncrna', 'TR_J_gene', 'TR_D_gene', 
#'IG_V_pseudogene', 'pseudogene', 'snRNA', 'unprocessed_pseudogene', 'TR_V_gene', 'transcribed_unprocessed_pseudogene', 'miRNA', 
#'translated_unprocessed_pseudogene', 'antisense', 'IG_C_gene', 'sense_overlapping', 'IG_D_gene', 'TR_C_gene', 'processed_transcript', 
#'transcribed_processed_pseudogene', 'snoRNA', 'processed_pseudogene'])

def main(bed_file, annotation_file, outfile):

    if not fileOk(bed_file) or len(annotation_file) <= 0:
        sys.stderr.write("Error, input file not present or invalid format\n")
        sys.exit(-1)
     
    if not outfile:
        outfile = "filtered_" + os.path.basename(bed_file)
           
    # loads all the coordinates to transcript type
    types = set()
    coord_to_type = dict()
    types_count = defaultdict(int)
    with open(annotation_file, "r") as filehandler:
        for line in filehandler.readlines()[5:]:
            tokens = line.split()
            chr = tokens[0]
            start = tokens[3]
            end = tokens[4]
            type = tokens[tokens.index('gene_biotype') + 1].split("\"")[1]
            types.add(type)
            coord_to_type[(chr,start,end)] = type
    print '\n'.join(x for x in types)   
       
    # writes entries that contain a barcode in the previous list
    with open(bed_file, "r") as filehandler_read:
        with open(outfile, "w") as filehandler_write:
            lines = filehandler_read.readlines()
            first_line = lines.pop(0).strip()
            filehandler_write.write(first_line)
            for line in lines:
                tokens = line.split()
                chr = tokens[7]
                start = tokens[7]
                end = tokens[7]
                for (chr_orig, start_orig, end_orig), type in coord_to_type.iteritems():
                    if chr == chr_orig and start >= start_orig and start <= end_orig:
                        types_count[type] += 1
                        if type in filter: filehandler_write.write(line)
    for type,count in types_counts.iteritems():
        print type + " " + str(count) + "\n"                         
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bed_file", 
                        help="Tab delimited file containing the clusters and the barcodes")
    parser.add_argument("--outfile", 
                        help="Name of the output file")
    parser.add_argument("--annotation-file",
                        help="The annotation file used in the pipeline (Ensemble)")
    args = parser.parse_args()
    main(args.bed_file, args.annotation_file, args.outfile)


