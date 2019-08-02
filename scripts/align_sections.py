import tissue_recognition as tr
import numpy as np
from imageio import imread
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.image import AxesImage
import cv2
from skimage import img_as_bool
from skimage.transform import resize
from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, plot_matches)
from skimage.exposure import cumulative_distribution
from skimage.viewer import ImageViewer
from skimage.viewer.canvastools import RectangleTool
from skimage.draw import line
from skimage.draw import set_color
import os
import pandas as pd
import re
import sys
import argparse
import scipy
from math import sqrt
from sklearn.neighbors import NearestNeighbors 
    
def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps 
    corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''
    assert A.shape == B.shape
    # get number of dimensions
    m = A.shape[1]
    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)
    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)
    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t
    return T, R, t

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''
    assert src.shape == dst.shape
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''
    assert A.shape == B.shape
    # get number of dimensions
    m = A.shape[1]
    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)
    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0
    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)
        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)
        # update the current source
        src = np.dot(T, src)
        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)
    return T, distances, i

def plot_images(image_list, is_gray=False, filename=None):
    columns = int(round(np.sqrt(len(image_list))))
    rows = int(np.ceil(np.sqrt(len(image_list))))
    fig = plt.figure(figsize = (16, 16))
    for i in range(1, columns*rows + 1):
        try:
            img = image_list[i - 1]
            ax = fig.add_subplot(rows, columns, i)
            if is_gray:
                plt.gray()
            ax.imshow(img)
        except:
            continue
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.clf()
    
def get_binary_masks(image_list):
    """Obtains a mask of the image using
    the tissue recognition library (automatic)
    """
    assert(len(image_list) > 0)
    binary_masks = list()
    for img in image_list:
        msk = np.zeros(img.shape[0:2], dtype=np.uint8)
        tr.recognize_tissue(img, msk)
        binary_masks.append(tr.get_binary_mask(msk))
    return binary_masks

def crop_mask_images(image_list):
    """Let the user select the mask for each
    image using a rectangle
    """
    assert(len(image_list) > 0)
    masked_cropped_images = list()
    for img in image_list:
        viewer = ImageViewer(img)
        rect_tool = RectangleTool(viewer)
        viewer.show()
        coord = np.int64(rect_tool.extents)
        h, w, z = img.shape
        crop_img = 255 - np.zeros((h, w, 3), dtype=np.uint8)
        crop_img[coord[2]:coord[3], coord[0]:coord[1]] = img[coord[2]:coord[3], coord[0]:coord[1]]
        masked_cropped_images.append(crop_img)
    return masked_cropped_images
    
def mask_images(image_list, binary_masks):
    """Applies binary masks to a list of images
    """
    assert(len(image_list) == len(binary_masks))
    masked_images = list()
    for img, bin_msk in zip(image_list, binary_masks):
        h, w, z = img.shape
        bin_msk = bin_msk > 0.5
        comb_img = 255 - np.zeros((h, w, 3), dtype=np.uint8)
        comb_img[bin_msk] = img[bin_msk]
        masked_images.append(comb_img)
    return masked_images

def mse(imageA, imageB):
    """The 'Mean Squared Error' between the two images is the
    sum of the squared difference between the two images;
    NOTE: the two images must have the same dimension
    returns the MSE, the lower the error, the more "similar"
    the two images are
    """
    assert(imageA.shape == imageB.shape)
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def euclidean_alignment(image_list, number_of_iterations=1000, termination_eps=1e-10):
    """ Aligns images using the first image as reference
    The methods uses the brigthness and the contour to find
    the best alignment.
    Returns the aligned images and the warp transformations
    """
    ref_img = image_list[0]
    warp_list = [None]
    aligned_images = [ref_img]
    
    # All images should have the same size
    sz = ref_img.shape
    
    # Some constants
    warp_mode = cv2.MOTION_EUCLIDEAN
    warp_matrix_orig = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                number_of_iterations, termination_eps)
        
    # Iterate images using first one as reference to align them
    for img in image_list[1:]:
        warp_matrix = None
        img_aligned = None
        best_mse = 10e6
        # Try the four reflections
        for (sx,sy) in [(1,1), (1,-1), (-1,1), (-1,-1)]:
            reflection = np.asanyarray([[sx, 0,  0],
                                        [0,  sy, 0]], np.float32)
            img_reflected = cv2.warpAffine(img, reflection, (sz[1], sz[0]),
                                           flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                           borderValue=(255, 255, 255))
            # Try the aligment                
            try:
                _, warp_matrix_new = cv2.findTransformECC(ref_img,
                                                          img_reflected,
                                                          warp_matrix_orig,
                                                          warp_mode,
                                                          criteria,
                                                          None,
                                                          5)
                warp_matrix_new = np.dot(np.vstack([reflection, [0, 0, 1]]), 
                                         np.vstack([warp_matrix_new, [0, 0, 1]]))
                warp_matrix_new = np.float32(np.delete(warp_matrix_new, (2), axis=0))
                img_aligned_new = cv2.warpAffine(img, warp_matrix_new, (sz[1], sz[0]),
                                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                                 borderValue=(255, 255, 255))
                # Compute alignment score to keep the best one
                new_mse = mse(ref_img, img_aligned_new)
                if new_mse < best_mse:
                    best_mseq = new_mse
                    img_aligned, warp_matrix = img_aligned_new, warp_matrix_new
            except Exception as e:
                pass
        if warp_matrix is None or img_aligned is None:
            print("Images could not be aligned properly")
            warp_matrix = np.asanyarray([[1,0,1], [0,1,1]])
            img_aligned = img
            
        # Store the aligned image and the transformation matrix
        warp_list.append(warp_matrix)
        aligned_images.append(img_aligned)
    return aligned_images, warp_list

def data_driven_aligment(image_list, counts_list):
    """ Aligns images using the first image as reference
    The methods selects points between the images
    based on similarities in the data (Spots)
    Returns the aligned images and the warp transformations
    """
    ref_img = image_list[0]
    aligned_images = [ref_img]
    warp_list = [None]
    ref_counts = counts_list[0]
    for counts, img in zip(counts_list[1:], image_list[1:]):
        height, width = img.shape
        cor_pairs = list()
        shared_genes = np.intersect1d(ref_counts.columns, counts.columns)
        filter_ref_counts = ref_counts.loc[:,shared_genes]
        filter_counts = counts.loc[:,shared_genes]
        filter_ref_counts = np.log1p(filter_ref_counts)
        filter_counts = np.log1p(filter_counts)
        for spot_ref in filter_ref_counts.index:
            x_ref, y_ref = spot_ref.split("x")
            best_x = -1
            best_y = -1
            best_corr = -1
            best_spot = (-1,-1)
            for spot in filter_counts.index:
                x, y = spot.split("x")
                corr, _ = scipy.stats.pearsonr(filter_ref_counts.loc[spot_ref,:].to_numpy(),
                                               filter_counts.loc[spot,:].to_numpy())
                if corr > best_corr:
                    best_x = x
                    best_y = y
                    best_corr = corr
                    best_spot = spot
            if best_spot is not (-1,-1):
                filter_counts.drop(best_spot,  inplace=True)
            cor_pairs.append(((float(x_ref), float(y_ref)),
                             (float(best_x), float(best_y)),
                             best_corr))
        sorted_cor_pairs = sorted(cor_pairs, key=lambda tup: tup[2], reverse=True)
        sx = width / 32
        sy = height / 34
        t = np.array([[sx, 0, -sx], [0, sy, -sy], [0, 0, 1]])
        ref_points_match = list()
        points_match = list()
        for ref, p, corr in sorted_cor_pairs[0:10]:
               ref_points_match.append([ref[0], ref[1]])  
               points_match.append([p[0], p[1]])
               
        ref_points_match = np.asanyarray(ref_points_match)
        ref_points_match = np.hstack([ref_points_match, np.ones((ref_points_match.shape[0], 1))])
        
        points_match = np.asanyarray(points_match)
        points_match = np.hstack([points_match, np.ones((points_match.shape[0], 1))])
        
        ref_points_match = np.dot(ref_points_match, t)
        ref_points_match = np.delete(ref_points_match, 2, 1)
        
        points_match = np.dot(points_match, t)
        points_match = np.delete(points_match, 2, 1)
        
        # Find transformation
        warp_matrix, _, _ = icp(points_match, ref_points_match)
        warp_matrix = np.float32(np.delete(warp_matrix, (2), axis=0))
    
        # Use transformation
        img_aligned = cv2.warpAffine(img, warp_matrix, (width, height),
                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                    borderValue=(255, 255, 255))
            
        
        if warp_matrix is None or img_aligned is None:
            print("Images could not be aligned properly")
            warp_matrix = np.asanyarray([[1,0,1], [0,1,1]])
            img_aligned = img
        
        # Store the aligned image and the transformation matrix
        warp_list.append(warp_matrix)
        aligned_images.append(img_aligned)
        
    return aligned_images, warp_list
                             
def homography_alignment(image_list):
    """ Aligns images using the first image as reference
    The methods selects points between the images
    and the reference image.
    Returns the aligned images and the warp transformations
    """
    ref_img = image_list[0]
    warp_list = [None]
    aligned_images = [ref_img]
    
    # All images should have the same size
    height, width = ref_img.shape
            
    MAX_FEATURES = 50
    GOOD_MATCH_PERCENT = 0.25
    orb = cv2.ORB_create(MAX_FEATURES)
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    keypoints2, descriptors2 = orb.detectAndCompute(ref_img, None)
    
    # Iterate images using first one as reference to align them
    for img in image_list[1:]:
        # Perform alignment
        keypoints1, descriptors1 = orb.detectAndCompute(img, None)
   
        # Match features.
        matches = matcher.match(descriptors1, descriptors2, None)
   
        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)
        
        # Remove not so good matches
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]
        
        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)
        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt
   
        # Find transformation
        warp_matrix_new, _, _ = best_fit_transform(points1, points2)
        warp_matrix_new = np.delete(warp_matrix_new, 2, 0)

         # Try the four reflections
        warp_matrix = None
        img_aligned = None
        best_mse = 10e6
        for (sx,sy) in [(1,1), (1,-1), (-1,1), (-1,-1)]:
            reflection = np.asanyarray([[sx, 0,  0],
                                        [0,  sy, 0]], np.float32)
            img_reflected = cv2.warpAffine(img, reflection, (width, height),
                                           flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                           borderValue=(255, 255, 255))
            # Combine reflection with transform
            warp_matrix_new = np.dot(np.vstack([reflection, [0, 0, 1]]), 
                                     np.vstack([warp_matrix_new, [0, 0, 1]]))
            warp_matrix_new = np.float32(np.delete(warp_matrix_new, (2), axis=0))
            
            # Use transformation
            img_aligned_new = cv2.warpAffine(img, warp_matrix_new, (width, height),
                                             flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                             borderValue=(255, 255, 255))
            
            # Compute alignment score to keep the best one
            new_mse = mse(ref_img, img_aligned_new)
            if new_mse < best_mse:
                best_mseq = new_mse
                img_aligned, warp_matrix = img_aligned_new, warp_matrix_new
                
        if warp_matrix is None or img_aligned is None:
            print("Images could not be aligned properly")
            warp_matrix = np.asanyarray([[1,0,1], [0,1,1]])
            img_aligned = img
        
        # Store the aligned image and the transformation matrix
        warp_list.append(warp_matrix)
        aligned_images.append(img_aligned)
        
    return aligned_images, warp_list

def project_warp_matrix(warp_list, images, w, h):
    """Adjusts warp transformation matrices to work on the 
    original images
    """
    assert(len(warp_list) == len(images))
    warp_list_backtransf = [None]
    for warp_matrix, img in zip(warp_list[1:], images[1:]):
        sz = img.shape
        scale_factor_x = sz[1] / w
        scale_factor_y = sz[0] / h
        project_down = np.array([[1 / scale_factor_x, 0, 1],
                                 [0, 1 / scale_factor_y, 1],
                                 [0, 0,  1]])
        project_up = np.linalg.inv(project_down)
        A = np.vstack([warp_matrix, [0, 0, 1]])
        B = np.dot(np.dot(project_up, A), project_down)
        B = np.delete(B, 2, 0)
        warp_list_backtransf.append(B)
    return warp_list_backtransf

def transform_original_image(image_list, warp_list_backtransf):
    """ Given a list of images and a list of affine matrices
    transform the images with the matrices and returns the 
    transformed images
    """
    assert(len(warp_list_backtransf) == len(image_list))
    transformed_images = [image_list[0]]
    # Run trough images and apply the transformation matrix
    for img, warp_matrix in zip(image_list[1:], warp_list_backtransf[1:]):
        sz = img.shape
        img_aligned = cv2.warpAffine(img, warp_matrix, (sz[1], sz[0]), 
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                     borderValue=(255, 255, 255))
        transformed_images.append(img_aligned)
    return transformed_images

def transform_counts(counts_list, image_list, warp_list_backtransf):
    """ Given a list of counts matrices and a list of affine matrices
    transform the spots in the matrices with the affine matrices and returns the 
    transformed counts matrices
    """
    assert(len(warp_list_backtransf) == len(image_list) == len(counts_list))
    transformed_counts = [counts_list[0]]
    # Run trough counts and apply the transformation matrix
    for counts, img, warp_matrix in zip(counts_list[1:], image_list[1:], warp_list_backtransf[1:]):
        height, width, _ = img.shape
        sx = width / 32
        sy = height / 34
        t = np.array([[sx, 0, -sx], [0, sy, -sy], [0, 0, 1]])
        t_inv = np.linalg.inv(t)
        spot_coords = np.zeros((counts.shape[0],2), dtype=np.uint8)
        for i,spot in enumerate(counts.index):
            x,y = spot.split("x")
            spot_coords[i,0] = float(x)
            spot_coords[i,1] = float(y)
        spot_coords = np.hstack([spot_coords, np.ones((spot_coords.shape[0], 1))])
        pixel_coords = np.dot(spot_coords, t)
        pixel_coords[:,2] = 1
        pixel_coords_adjusted = np.dot(cv2.invertAffineTransform(warp_matrix), pixel_coords.transpose()).transpose()
        pixel_coords_adjusted = np.hstack([pixel_coords_adjusted, np.ones((pixel_coords_adjusted.shape[0], 1))])
        spot_coords_adjusted = np.dot(pixel_coords_adjusted, t_inv)
        counts.index = ["{}x{}".format(abs(x), abs(y)) for x,y in zip(np.around(spot_coords_adjusted[:,0], 2),
                                                                      np.around(spot_coords_adjusted[:,1], 2))]
        transformed_counts.append(counts)
    return transformed_counts

def hist_norm(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image
    """
    olddtype = source.dtype
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and counts
    s_values, bin_idx, s_counts = np.unique(source, 
                                            return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)
    
    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
    
    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    interp_t_values = interp_t_values.astype(olddtype)

    return interp_t_values[bin_idx].reshape(oldshape)

def main(counts_files, images_files, down_width, down_height, outdir,
         debug, manual, alignment_method, normalize):
    
    if len(counts_files) == 0 or \
    any([not os.path.isfile(f) for f in counts_files]):
        sys.stderr.write("Error, input counts matrices not present or invalid format\n")
        sys.exit(1)
        
    if len(images_files) == 0 or \
    any([not os.path.isfile(f) for f in images_files]):
        sys.stderr.write("Error, input images not present or invalid format\n")
        sys.exit(1)
    
    if len(images_files) != len(counts_files):
        sys.stderr.write("Error, counts and images have different size\n")
        sys.exit(1)
            
    if down_width < 100 or down_height < 100:
        sys.stderr.write("Error, invalid scaling factors\n")
        sys.exit(1)
        
    if outdir is None or not os.path.isdir(outdir): 
        outdir = os.getcwd()
    outdir = os.path.abspath(outdir)

    print("Output directory {}".format(outdir))
    print("Input datasets {}".format(" ".join(counts_files)))
    print("Input images {}".format(" ".join(images_files))) 
    
    print("Parsing counts matrices...")
    counts = [pd.read_csv(c, sep="\t", header=0,
                          index_col=0, engine='c', low_memory=True) for c in counts_files]
    
    print("Parsing images and resizing...")
    images = [cv2.cvtColor(cv2.imread(file), 
                           cv2.COLOR_BGR2RGB) for file in images_files]
    images_resized = [cv2.resize(img, (down_width, down_height), 
                                 interpolation=cv2.INTER_CUBIC) for img in images]
    if debug:
        plot_images(images_resized, False, os.path.join("images_resized.png"))
        
    print("Computing masks for the images...")
    if not manual:
        binary_masks = get_binary_masks(images_resized)
        masked_images = mask_images(images_resized, binary_masks)
    else:
        masked_images = crop_mask_images(images_resized)
    if debug:
        plot_images(masked_images, False, os.path.join(outdir, "masked_images.png"))
    
    print("Performing alignment...")
    grey_masked_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in masked_images]
    grey_masked_images = [cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX) for img in grey_masked_images]
    if normalize:
        grey_masked_images = [grey_masked_images[0]] + \
                             [hist_norm(img, grey_masked_images[0]) for img in grey_masked_images[1:]]
    if debug:
        plot_images(grey_masked_images, True, os.path.join("grey_masked_images.png"))
    
    if alignment_method == "HOMO":
        aligned_images, warp_list = homography_alignment(grey_masked_images)
    elif alignment_method == "DATA":
        aligned_images, warp_list = data_driven_aligment(grey_masked_images, counts)
    else:
        aligned_images, warp_list = euclidean_alignment(grey_masked_images)
    if debug:
        plot_images(aligned_images, True, os.path.join("aligned_grey_images.png"))
        
    print("Transforming original images and counts matrices...")
    warp_list_backtransf = project_warp_matrix(warp_list, images, down_width, down_height)
    transformed_original_images = transform_original_image(images, warp_list_backtransf)
    if debug:
        plot_images(transformed_original_images, False, os.path.join("aligned_images.png"))
    
    # Transform counts matrices spots
    transformed_counts = transform_counts(counts, images, warp_list_backtransf)
    
    # Save images
    for img, name in zip(transformed_original_images, images_files):
        matplotlib.image.imsave(os.path.join(outdir, "aligned_" + name), img)
    
    # Save counts
    for counts, name in zip(transformed_counts, counts_files):
        counts.to_csv(os.path.join(outdir, "aligned_" + name), sep="\t")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--counts", required=True, nargs='+', type=str,
                        help="One or more matrices with gene counts per feature/spot (genes as columns)")
    parser.add_argument("--images", required=True, nargs='+', type=str,
                        help="The HE images corresponding to the counts matrices (same order)")
    parser.add_argument("--down-width", default=500, metavar="[INT]", type=int,
                        help="The size of the width in pixels of the down-sampled images (default: %(default)s)")
    parser.add_argument("--down-height", default=500, metavar="[INT]", type=int,
                        help="The size of the height in pixels of the down-sampled images (default: %(default)s)")
    parser.add_argument("--outdir", default=None, help="Path to output dir")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Whether to generate debug images for each step")
    parser.add_argument("--manual", action="store_true", default=False,
                        help="Whether to perform a manual selection to mask each image, otherwise it will be done automatically")
    parser.add_argument("--normalize", action="store_true", default=False,
                        help="Whether to perform a cumulative histogram normalization on the images to equalize the intensities")
    parser.add_argument("--alignment-method", default="ECC", metavar="[STR]", 
                        type=str, 
                        choices=["ECC", "HOMO",  "DATA"],
                        help="The algorithm to use for the image alignment:\n" \
                        "ECC = automatic alignment based on the brightness and contour (ECC algorithm) \n" \
                        "HOMO = automatic alignment based on feature extractions (points) (Homographic) \n" \
                        "DATA = alignment based on the expression data (spots similarities as reference points)\n" \
                        "(default: %(default)s)")
    args = parser.parse_args()
    main(args.counts, args.images, args.down_width, args.down_height, args.outdir,
         args.debug, args.manual, args.alignment_method, args.normalize)