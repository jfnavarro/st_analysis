#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script that converts a Visium 10x dataset
to a Spatial Transcriptomics dataset
"""
import os
import scipy.io
import pandas as pd
import numpy as np
import sys
import argparse
import json


# Most of this code was borrowed from Alma Anderson (github@almaan)
def main(matrix, features, barcodes, positions, outdir, scale_factors, adjust):

    if not os.path.isfile(matrix):
        sys.stderr.write("Error, input matrix not present or invalid format\n")
        sys.exit(1)

    if not os.path.isfile(features):
        sys.stderr.write("Error, input features not present or invalid format\n")
        sys.exit(1)
        barcodes

    if not os.path.isfile(barcodes):
        sys.stderr.write("Error, input barcodes not present or invalid format\n")
        sys.exit(1)

    if not os.path.isfile(positions):
        sys.stderr.write("Error, input positions not present or invalid format\n")
        sys.exit(1)

    if not os.path.isfile(scale_factors):
        sys.stderr.write("Error, scale factors not present or invalid format\n")
        sys.exit(1)

    if not outdir:
        outdir = os.getcwd()
    print("Output folder {}".format(outdir))

    # Parse the genes
    genes = pd.read_csv(features, sep='\t', header=None, index_col=None)
    genes.columns = ['ensg', 'hgnc', 'info']
    genes = genes['hgnc'].values

    # Parse the barcodes
    barcodes = pd.read_csv(barcodes, sep='\t',
                           header=None, index_col=None).values.flatten()

    # Parse the matrix of counts
    mat = scipy.io.mmread(matrix)

    # Parse the positions and create the ST data frame
    dmat = pd.DataFrame(mat.toarray().T, index=barcodes, columns=genes)
    tp = pd.read_csv(positions, sep=',', header=None, index_col=0)
    tp.columns = ['under', 'x', 'y', 'px', 'py']
    tp = tp.loc[tp['under'] == 1, :]
    inter = tp.index.intersection(dmat.index)
    tp = tp.loc[inter, :]
    dmat = dmat.loc[inter, :]
    dmat.index = ["{}x{}".format(x, y) for x, y in zip(tp['x'].values, tp['y'].values)]
    dmat.to_csv(os.path.join(outdir, "st_data.tsv"), index=True, header=True, sep='\t')

    # Load scale factor
    with open(scale_factors) as json_file:
        sf = float(json.load(json_file)['tissue_hires_scalef'])

    # Create the ST spot coordinate file (Pixel x and y coordinates are transposed in Visium)
    tp.index = dmat.index
    tp = tp.loc[:, ['py', 'px']]
    tp = tp * sf
    if adjust:
        tp = pd.DataFrame(data=np.transpose(np.rot90(tp, k=1, axes=(1, 0))),
                          index=tp.index,
                          columns=tp.columns)
    tp.to_csv(os.path.join(outdir, "st_coordinates.tsv"), index=True, header=False, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", required=True, type=str,
                        help="Visium matrix file in GMX format")
    parser.add_argument("--features", required=True, type=str,
                        help="Visium features file in TSV format")
    parser.add_argument("--barcodes", required=True, type=str,
                        help="Visium barcodes file in TSV format")
    parser.add_argument("--positions", required=True, type=str,
                        help="Visium positions file in CSV format")
    parser.add_argument("--scale-factors", required=True, type=str,
                        help="Visium scale factors in JSON format")
    parser.add_argument("--adjust", action="store_true", default=False,
                        help="Transform the Visium coordinates origin to ST coordinates origin")
    parser.add_argument("--outdir", help="Name of the output folder")
    args = parser.parse_args()
    main(args.matrix, args.features, args.barcodes, args.positions, args.outdir, args.scale_factors, args.adjust)
