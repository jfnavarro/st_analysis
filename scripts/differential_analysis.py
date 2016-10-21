#! /usr/bin/env python
""" 
This script performs Differential
Expression Analysis using DESeq2 
on a table with gene counts in the following format:

      GeneA   GeneB   GeneC
1x1   
1x2
...

The script also requires a file where spots are mapped
to a class (output from unsupervised.py). This
file is a tab delimited file like this:

CLASS SPOT 

The script also requires a 
list of classes to perform differential expression
analysis. For example 1-2 or 1-3, etc..

The script will output the list of up-regulated and down-regulated
for each DEA comparison as well as a set of plots.

@Author Jose Fernandez Navarro <jose.fernandez.navarro@scilifelab.se>
"""
import argparse
import sys
import os
import numpy as np
import pandas as pd
from stanalysis.normalization import RimportLibrary
from stanalysis.visualization import scatter_plot
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
   
def dea(counts, conds):
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
    dds = r.DESeq(dds)
    results = r.results(dds)
    results = pandas2ri.ri2py_dataframe(r['as.data.frame'](results))
    results.index = counts.index
    # Return the DESeq2 DEA results object
    pandas2ri.deactivate()
    return results
              
def main(input_data, data_classes, conditions_tuples, outdir):

    if not os.path.isfile(input_data) or not os.path.isfile(data_classes) \
    or len(conditions_tuples) < 1:
        sys.stderr.write("Error, input file/s not present or invalid format\n")
        sys.exit(1)
     
    if not outdir or not os.path.isdir(outdir):
        outdir = os.getcwd()
        
    print "Output folder {}".format(outdir)
      
    # Spots are rows and genes are columns
    counts = pd.read_table(input_data, sep="\t", header=0, index_col=0)
    
    # loads all the classes for the spots
    spot_classes = dict()
    with open(data_classes) as filehandler:
        for line in filehandler.readlines():
            tokens = line.split()
            assert(len(tokens) == 2)
            spot_classes[tokens[1]] = str(tokens[0])
    assert(len(spot_classes) == len(counts.index))       
         
    # Iterate the conditions
    for cond in conditions_tuples:
        tokens = cond.split("-")
        assert(len(tokens) == 2)
        a = str(tokens[0])
        b = str(tokens[1])
        conds = list()
        # Genes as rows
        sub_counts = counts.transpose()
        for spot in sub_counts.columns:
            try:
                spot_class = spot_classes[spot]
                if spot_class in [a,b]:
                    conds.append(spot_class)
                elif b == "REST":
                    conds.append(b)
                else:
                    sub_counts.drop(spot, axis=1, inplace=True)
            except KeyError:
                sub_counts.drop(spot, axis=1, inplace=True)
        # Make the DEA call
        print "Doing DEA for the conditions {} ...".format(cond)
        dea_results = dea(sub_counts, conds)
        dea_results.sort_values(by=["pvalue"], ascending=True, inplace=True, axis=0)
        print "Writing results to output..."
        dea_results.to_csv(os.path.join(outdir, "dea_results_{}.tsv".format(cond)), sep="\t")
        # Volcano plot
        print "Generating plots..."
        scatter_plot(dea_results["log2FoldChange"], -np.log10(dea_results["pvalue"]),
                     xlabel="Log2FoldChange", ylabel="-log10(pvalue)", 
                     title="Volcano plot", output=os.path.join(outdir, "volcano_{}.png".format(cond)))
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-data",
                        help="The matrix of counts (spots as row names and genes as column names)")
    parser.add_argument("--data-classes", required=True,
                        help="A tab delimited file with the classes mapping to the spots " \
                        "(Class first column and spot second column)")
    parser.add_argument("--conditions-tuples", required=True, nargs='+', type=str,
                        help="One of more tuples that represent what classes will be compared for DEA, " \
                        "for example 1-2 1-3 2-REST")
    parser.add_argument("--outdir", help="Path to output dir")
    args = parser.parse_args()
    main(args.input_data, args.data_classes, args.conditions_tuples, args.outdir)


