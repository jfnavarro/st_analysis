#! /usr/bin/env python
""" 
This script performs Differential Expression Analysis 
using DESeq2 on a table with gene counts with the following format:

      GeneA   GeneB   GeneC
1x1   
1x2
...

The script can take one or several datasets.
The script also requires a file where spots are mapped
to a class for each dataset. 
This file is a tab delimited file like this:

CLASS SPOT 

The script also requires a 
list of classes/groups to perform differential expression
analysis. For example 1-2 or 1-3, etc..Where 1,2,3 are classes
defined in the tab delimited file.

The script will output the list of up-regulated and down-regulated
for each DEA comparison as well as a set of plots.

The script allows to normalize the data with different methods.

@Author Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>
"""
import argparse
import sys
import os
import numpy as np
import pandas as pd
from stanalysis.normalization import RimportLibrary
from stanalysis.visualization import scatter_plot
from stanalysis.normalization import *
from stanalysis.preprocessing import *
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, r, globalenv

def get_classes_coordinate(class_file):
    """ Helper function
    to get a dictionary of spot -> class 
    from a tab delimited file
    """
    barcodes_classes = dict()
    with open(class_file, "r") as filehandler:
        for line in filehandler.readlines():
            tokens = line.split()
            assert(len(tokens) == 2)
            spot = tokens[1]
            class_label = tokens[0]
            barcodes_classes[spot] = class_label
    return barcodes_classes
   
def dea(counts, conds, size_factors=None):
    """Makes a call to DESeq2 to
    perform D.E.A. in the given
    counts matrix with the given conditions
    """
    pandas2ri.activate()
    deseq2 = RimportLibrary("DESeq2")
    r("suppressMessages(library(DESeq2))")
    # Create the R conditions and counts data
    r_counts = pandas2ri.py2ri(counts)
    cond = robjects.DataFrame({"conditions": robjects.StrVector(conds)})
    design = r('formula(~ conditions)')
    dds = r.DESeqDataSetFromMatrix(countData=r_counts, colData=cond, design=design)
    if size_factors is None:
        dds = r.DESeq(dds)
    else:
        assign_sf = r["sizeFactors<-"]
        dds = assign_sf(object=dds, value=robjects.FloatVector(size_factors))
        dds = r.estimateDispersions(dds)
        dds = r.nbinomWaldTest(dds)
    results = r.results(dds, contrast=r.c("condition", "A", "B"))
    results = pandas2ri.ri2py_dataframe(r['as.data.frame'](results))
    results.index = counts.index
    # Return the DESeq2 DEA results object
    pandas2ri.deactivate()
    return results
              
def main(counts_table_files, data_classes, 
         conditions_tuples, outdir, fdr, normalization):

    if len(counts_table_files) == 0 or \
    any([not os.path.isfile(f) for f in counts_table_files]):
        sys.stderr.write("Error, input file/s not present or invalid format\n")
        sys.exit(1)
        
    if len(data_classes) == 0 or \
    any([not os.path.isfile(f) for f in counts_table_files]):
        sys.stderr.write("Error, input file/s not present or invalid format\n")
        sys.exit(1)
        
    if len(data_classes) != len(counts_table_files):
        sys.stderr.write("Error, input file/s not present or invalid format\n")
        sys.exit(1)
     
    if not outdir or not os.path.isdir(outdir):
        outdir = os.getcwd()
        
    print "Output folder {}".format(outdir)
      
    # Merge input datasets (Spots are rows and genes are columns)
    counts = aggregate_datatasets(counts_table_files)
    
    # loads all the classes for the spots
    spot_classes = dict()
    for i,class_file in enumerate(data_classes):
        with open(class_file) as filehandler:
            for line in filehandler.readlines():
                tokens = line.split()
                assert(len(tokens) == 2)
                spot_classes["{}_{]".format(i,tokens[1])] = str(tokens[0])  
     
    # Compute size factors
    size_factors = compute_size_factors(counts, normalization)
    
    # Spots as columns
    counts = counts.transpose()
    
    # Iterate the conditions
    for cond in conditions_tuples:
        new_counts = counts.copy()
        tokens = cond.split("-")
        assert(len(tokens) == 2)
        tokens_a = tokens[0].split(":")
        assert(len(tokens_a) == 2)
        tokens_b = tokens[1].split(":")
        assert(len(tokens_b) == 2)
        dataset_a = str(tokens_a[0])
        dataset_b = str(tokens_b[0])
        region_a = str(tokens_a[1])
        region_b = str(tokens_b[1])
        conds = list()
        for spot in new_counts.columns:
            try:
                spot_class = spot_classes[spot]
                if spot_class == a:
                    conds.append("A")
                elif spot_class == b:
                    conds.append("B")
                else:
                    new_counts.drop(spot, axis=1, inplace=True)
            except KeyError:
                new_counts.drop(spot, axis=1, inplace=True)
        # Make the DEA call
        print "Doing DEA for the conditions {} ...".format(cond)
        dea_results = dea(new_counts, conds, size_factors)
        dea_results.sort_values(by=["padj"], ascending=True, inplace=True, axis=0)
        print "Writing results to output..."
        dea_results.to_csv(os.path.join(outdir, "dea_results_{}.tsv".format(cond)), sep="\t")
        # Volcano plot
        print "Generating plots..."
        # Add colors according to differently expressed or not (needs a p-value parameter)
        colors = ["blue" if p <= fdr else "red" for p in dea_results["padj"]]
        scatter_plot(dea_results["log2FoldChange"], -np.log10(dea_results["pvalue"]),
                     xlabel="Log2FoldChange", ylabel="-log10(pvalue)", colors=colors,
                     title="Volcano plot", output=os.path.join(outdir, "volcano_{}.png".format(cond)))
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--counts-table-files", required=True, nargs='+', type=str,
                        help="One or more matrices with gene counts per feature/spot (genes as columns)")
    parser.add_argument("--data-classes", required=True, nargs='+', type=str,
                        help="One or more delimited file/s with the classes mapping to the spots " \
                        "(Class first column and spot second column)")
    parser.add_argument("--normalization", default="DESeq2", metavar="[STR]", 
                        type=str, 
                        choices=["RAW", "DESeq2", "DESeq2Linear", "DESeq2PseudoCount", 
                                 "DESeq2SizeAdjusted", "REL", "TMM", "RLE", "Scran"],
                        help="Normalize the counts using:\n" \
                        "RAW = absolute counts\n" \
                        "DESeq2 = DESeq2::estimateSizeFactors(counts)\n" \
                        "DESeq2PseudoCount = DESeq2::estimateSizeFactors(counts + 1)\n" \
                        "DESeq2Linear = DESeq2::estimateSizeFactors(counts, linear=TRUE)\n" \
                        "DESeq2SizeAdjusted = DESeq2::estimateSizeFactors(counts + lib_size_factors)\n" \
                        "RLE = EdgeR RLE * lib_size\n" \
                        "TMM = EdgeR TMM * lib_size\n" \
                        "Scran = Deconvolution Sum Factors (Marioni et al)\n" \
                        "REL = Each gene count divided by the total count of its spot\n" \
                        "(default: %(default)s)")
    parser.add_argument("--conditions-tuples", required=True, nargs='+', type=str,
                        help="One of more tuples that represent what classes and datasets will be compared for DEA, " \
                        "for example 0:1-1:2 1:1-1:3 0:2-0:1")
    parser.add_argument("--fdr", type=float, default=0.05,
                        help="The FDR minimum confidence threshold (default: %(default)s)")
    parser.add_argument("--outdir", help="Path to output dir")
    args = parser.parse_args()
    main(args.counts_table_files, args.data_classes, args.conditions_tuples, 
         args.outdir, args.fdr, args.normalization)