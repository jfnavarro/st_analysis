#! /usr/bin/env python
"""
Scripts that converts spot coordinates file without pixel coordinates
to the ST Spot detector format that contains the pixel coordinates

Input: HE image and spot_coordinates in the following format

SPOT_X SPOT_Y ARRAY_X ARRAY_Y

Output: spot_coordinates in the following format:

SPOT_X SPOT_Y ARRAY_X ARRAY_Y PIXEL_X PIXEL_Y

@Author Jose Fernandez Navarro <jc.fernandez.navarro@gmail.com>
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
import imageio

def main(image_file, coordinates_file, outfile):

    if not os.path.isfile(image_file):
        sys.stderr.write("Error, input image file not present or invalid format\n")
        sys.exit(1)

    if not os.path.isfile(coordinates_file):
        sys.stderr.write("Error, input spot coordinates file not present or invalid format\n")
        sys.exit(1)

    if not outfile:
        outfile = "converted_{}.tsv".format(os.path.basename(coordinates_file).split(".")[0])

    image = imageio.imread(image_file)
    coords = pd.read_csv(coordinates_file, sep='\t', header=None, index_col=None)
    if coords.shape[1] != 4:
        sys.stderr.write("Error, input spot coordinates file has the wrong format\n")
        sys.exit(1)
    coords.columns = ["spot_x", "spot_y", "array_x", "array_y"]
    height, width, _ = image.shape
    sx = width / 32
    sy = height / 34
    t = np.array([[sx, 0, 0],
                  [0, sy, 0],
                  [-sx, -sy, 1]])
    spot_coords = coords.loc[:, ["array_x", "array_y"]]
    spot_coords = np.hstack([spot_coords, np.ones((spot_coords.shape[0], 1))])
    pixel_coords = spot_coords @ t
    coords["pixel_x"] = pixel_coords[:,0]
    coords["pixel_y"] = pixel_coords[:,1]
    coords.to_csv(outfile, sep="\t", header=False, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--image", required=True, type=str,
                        help="The HE image corresponding to the dataset")
    parser.add_argument("--coordinates", required=True, type=str,
                        help="The spot coordinates file corresponding to the dataset")
    parser.add_argument("--outfile", default=None, help="Name of output file")

    args = parser.parse_args()
    main(args.image, args.coordinates, args.outfile)