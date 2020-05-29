#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Script that converts a Visium 10x dataset
into a Spatial Transcriptomics dataset
"""
import csv
import gzip
import os
import scipy.io
import numpy as np
import pandas as pd
import loompy as lp
import sys

def main(matrix, features, barcodes, positions, outdir):
    
    if not outdir:
        outfile = os.getcwd()
    print("Output file {}".format(outfile))
    
    if not os.path.isfile(matrix):
        sys.stderr.write("Error, input matrix not present or invalid format\n")
        sys.exit(1)

    if not os.path.isfile(features):
        sys.stderr.write("Error, input feaatures not present or invalid format\n")
        sys.exit(1)
        barcodes
        
    if not os.path.isfile(barcodes):
        sys.stderr.write("Error, input barcodes not present or invalid format\n")
        sys.exit(1)

    if not os.path.isfile(positions):
        sys.stderr.write("Error, input positions not present or invalid format\n")
        sys.exit(1)
        
        
    # Parse the matrix of counts
    mat = scipy.io.mmread(matrix)
    
    # Parse the genes
    genes = pd.read_csv(barcodes, sep='\t', header=None, index_col=None)
    genes.columns = pd.Index(['ensg', 'hgnc', 'info'])
    genes = genes['hgnc'].values
    
    # Parse the barcodes
    barcodes = pd.Index(pd.read_csv(barcodes, sep='\t', 
                                    header=None, index_col=None).values.reshape(-1,))
    
    # Parse the positions and create the ST data frame
    dmat = pd.DataFrame(mat.toarray().T, index=barcodes, columns=genes)
    tp = pd.read_csv(positions, sep=',', header=None, index_col=0)
    tp.columns = ['under', 'x', 'y', 'px', 'py']
    tp = tp.loc[tp['under'] == 1,:]
    inter = tp.index.intersection(dmat.index)
    tp = tp.loc[inter,:]
    dmat = dmat.loc[inter,:]
    dmat.index  = ["{}x{}".format(x,y) for x,y in zip(tp['x'].values, tp['y'].values)]
    dmat.to_csv(os.path.join(outdir, "st_data.tsv"), index=True, header=True, sep='\t')
    
    # Create the ST spot coordinate file
    tp = to.loc[:,['x', 'y', 'px', 'py']]
    tp.index = dmat.index
    tp.to_csv(os.path.join(outdir, "st_coordinates.tsv"), index=True, header=False, sep='\t')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", required=True, type=str,
                        help="10x matrix file in GMX format")
    parser.add_argument("--features", required=True, type=str,
                        help="10x features file in TSV format")
    parser.add_argument("--barcodes", action="store_true", default=False,
                        help="10x barcodes file in TSV format")
    parser.add_argument("--positions", action="store_true", default=False,
                        help="10x positions file in CSV format")
    parser.add_argument("--outfile", help="Name of the output file")
    args = parser.parse_args()
    main(args.matrix, args.features, args.barcodes, args.positions, args.outfile)
