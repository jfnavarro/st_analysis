#! /usr/bin/env python
"""
Script that stiches a set of Spatial Transcriptomics datasets
together to create one single image/counts matrix/spot coordinates.
The (i,j) position of the dataset in the stiched image is appended
to the spots ids.

@Author Jose Fernandez Navarro <jc.fernandez.navarro@gmail.com>
"""

import argparse
import sys
import os
import pandas as pd
import imageio
import numpy as np
from skimage.transform import rescale, resize

def main(counts_files, images_files, coordinates_files, down_width, down_height, outdir, num_columns):

    if len(counts_files) == 0 or \
            any([not os.path.isfile(f) for f in counts_files]):
        sys.stderr.write("Error, input counts matrices not present or invalid format\n")
        sys.exit(1)

    if len(images_files) == 0 or \
            any([not os.path.isfile(f) for f in images_files]):
        sys.stderr.write("Error, input images not present or invalid format\n")
        sys.exit(1)

    if len(coordinates_files) == 0 or \
            any([not os.path.isfile(f) for f in coordinates_files]):
        sys.stderr.write("Error, input spot coordinates not present or invalid format\n")
        sys.exit(1)

    if len(images_files) != len(counts_files) or len(coordinates_files) != len(counts_files):
        sys.stderr.write("Error, counts and images or spot coordinates have different sizes\n")
        sys.exit(1)

    if down_width < 100 or down_height < 100:
        sys.stderr.write("Error, invalid scaling factors (too small)\n")
        sys.exit(1)

    if num_columns > len(counts_files) or num_columns <= 0:
        sys.stderr.write("Error, invalid number of columns\n")
        sys.exit(1)

    if outdir is None or not os.path.isdir(outdir):
        outdir = os.getcwd()
    outdir = os.path.abspath(outdir)

    # Compute number of columns/rows
    n_col = min(num_columns, len(counts_files))
    n_row = max(int(len(counts_files) / n_col), 1)
    # Stiched image/counts/coordinates
    img_stitched = np.zeros((down_height * n_row, down_width * n_col, 3), dtype=np.double)
    pixel_coords_stiched = np.empty((0,2))
    stiched_counts = pd.DataFrame()
    i = 0
    j = 0
    for counts_file, img_file, coord_file in zip(counts_files, images_files, coordinates_files):
        counts = pd.read_csv(counts_file, sep="\t", header=0, index_col=0)
        spots = pd.read_csv(coord_file, sep="\t", header=0, index_col=None)
        if spots.shape[1] == 7:
            spots = spots.loc[spots.iloc[:,6] == 1]
        spots.index = ["{}x{}".format(x,y) for x,y in zip(spots.iloc[:,0], spots.iloc[:,1])]
        img = imageio.imread(img_file)

        shared_spots = np.intersect1d(counts.index, spots.index)
        counts = counts.loc[shared_spots,:]
        spots = spots.loc[shared_spots,:]

        counts.index = ["{}{}_{}".format(i, j, spot) for spot in counts.index]
        stiched_counts = stiched_counts.append(counts, sort=True)

        height, width, _ = img.shape
        offset_x = i * down_width
        offset_y = j * down_height

        # Can probably combine these two transformations into one
        t1 = np.array([[1, 0, offset_x],
                       [0, 1, offset_y],
                       [0, 0, 1]])

        t2 = np.array([[down_width / width, 0, 0],
                       [0, down_height / height, 0],
                       [ 0, 0, 1]])

        pixel_coords = spots.iloc[:, [4, 5]]
        pixel_coords = np.hstack([pixel_coords, np.ones((pixel_coords.shape[0], 1))])
        pixel_coords = pixel_coords @ t2 @ t1
        pixel_coords_stiched = np.vstack([pixel_coords_stiched, pixel_coords[:,[0,1]]])

        img_resized = resize(img, (down_height, down_width), anti_aliasing=True)
        img_stitched[offset_y:offset_y + down_height, offset_x:offset_x + down_width, :] = img_resized

        i += 1
        if i == n_col:
            i = 0
            j += 1

    # Replace Nan and Inf by zeroes
    stiched_counts.replace([np.inf, -np.inf], np.nan)
    stiched_counts.fillna(0.0, inplace=True)
    stiched_counts.to_csv(os.path.join(outdir, "stiched_counts.tsv"), sep="\t")

    pd.DataFrame(data=pixel_coords_stiched, index=stiched_counts.index, columns=None).to_csv(
        os.path.join(outdir, "stiched_coordinates.tsv"), sep="\t", header=False, index=True)

    imageio.imwrite(os.path.join(outdir, "stiched_he.jpg"), img_stitched)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--counts", required=True, nargs='+', type=str,
                        help="One or more matrices with gene counts per spot (genes as columns)")
    parser.add_argument("--images", required=True, nargs='+', type=str,
                        help="The HE images corresponding to the counts matrices (same order)")
    parser.add_argument("--coordinates", required=True, nargs='+', type=str,
                        help="The spot coordinates files corresponding to the counts matrices (same order)")
    parser.add_argument("--down-width", default=2000, metavar="[INT]", type=int,
                        help="The size if pixels of the downsampled images (width) (default: %(default)s)")
    parser.add_argument("--down-height", default=2000, metavar="[INT]", type=int,
                        help="The size if pixels of the downsampled images  (height) (default: %(default)s)")
    parser.add_argument("--columns", default=None, metavar="[INT]", type=int, required=True,
                        help="The number of columns for the stiched image")
    parser.add_argument("--outdir", default=None, help="Path to output dir")

    args = parser.parse_args()
    main(args.counts, args.images, args.coordinates,
         args.down_width, args.down_height, args.outdir, args.columns)