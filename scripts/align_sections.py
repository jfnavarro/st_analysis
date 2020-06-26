#! /usr/bin/env python
""" 
This script aligns a set of images/counts matrices using the first
image as reference. It provides different alignment methods (manual
and automatic) and it outputs the aligned images, the aligned counts
matrices and the alignment matrices.

@TODO remove spots outside frame when converted
@TODO allow to pass spot coordinates files and transform it too

@Author Jose Fernandez Navarro <jc.fernandez.navarro@gmail.com>
"""
try:
    import tissue_recognition as tr
except ImportError as e:
    print("Error importing st_tissue_recognition")
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.image import AxesImage
import cv2
import os
import pandas as pd
import sys
import argparse
from sklearn.neighbors import NearestNeighbors
from skimage.measure import regionprops
from skimage import filters

def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps 
    corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    """
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
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    """
    assert src.shape == dst.shape
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def icp(A, B, init_pose=None, max_iterations=50, tolerance=0.001):
    """
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
    """
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
    """
    Helper function to debug
    Args:
        image_list: list of images (OpenCV)
        is_gray: whether the images are in grayscale or not
        filename: name of the output file

    Returns: None

    """
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
    plt.cla()
    
def get_binary_masks(image_list):
    """
    Obtains a mask of the image using
    the tissue recognition library (automatic)
    """
    assert(len(image_list) > 0)
    binary_masks = list()
    for img in image_list:
        msk = np.zeros(img.shape[0:2], dtype=np.uint8)
        tr.recognize_tissue(img, msk)
        binary_masks.append(tr.get_binary_mask(msk))
    return binary_masks
    
def mask_images(image_list, binary_masks):
    """
    Applies binary masks to a list of images
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

def center_images(image_list, binary_masks):
    """
    Centers masked images using their center of mass
    Returns the centered images and the transformations
    """
    assert(len(image_list) == len(binary_masks))
    centered_images = list()
    centered_maks = list()
    transf_matrices = list()
    for image, bin_msk in zip(image_list, binary_masks):
        h, w, _ = image.shape
        threshold_value = filters.threshold_otsu(bin_msk)
        labeled_foreground = (bin_msk > threshold_value).astype(int)
        properties = regionprops(labeled_foreground, bin_msk)
        center_of_mass = properties[0].centroid
        translation = np.float32([[1, 0, (w / 2) - center_of_mass[1]],
                                  [0, 1, (h / 2) - center_of_mass[0]]])
        img_centered = cv2.warpAffine(image, translation, (w, h), 
                                      borderValue=(255, 255, 255))
        mask_centered = cv2.warpAffine(bin_msk, translation, (w, h), 
                                      borderValue=(0, 0, 0))
        centered_images.append(img_centered)
        transf_matrices.append(translation)
        centered_maks.append(mask_centered)
    return centered_images, centered_maks, transf_matrices

def add_borders(image_list, size=1000):
    """
    Adds a white border to images
    """
    images_border = list()
    for image in image_list:
        border_image = cv2.copyMakeBorder(image, 
                                          size, size, size, size,
                                          cv2.BORDER_CONSTANT,
                                          value=(255, 255, 255))
        images_border.append(border_image)
    return images_border

def remove_borders(image_list, size=1000):
    """
    Removes a white border of images
    """
    images_no_border = list()
    for image in image_list:
        h, w, _ = image.shape
        no_border_image = image[size:h-size, size:w-size]
        images_no_border.append(no_border_image)
    return images_no_border
    
def mse(imageA, imageB):
    """
    The 'Mean Squared Error' between the two images is the
    sum of the squared difference between the two images;
    NOTE: the two images must have the same dimension
    returns the MSE, the lower the error, the more "similar"
    the two images are
    """
    assert(imageA.shape == imageB.shape)
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def euclidean_alignment(image_list, number_of_iterations=1000, termination_eps=0.0001):
    """
    Aligns images using the first image as reference
    The methods uses the brightness and the contour to find
    the best alignment.
    Returns the aligned images and the warp transformations
    """
    ref_img = image_list[0]
    warp_list = [np.eye(2, 3, dtype=np.float32)]
    aligned_images = [ref_img]
    
    # Some constants
    warp_matrix_orig = np.eye(2, 3, dtype=np.float32)
    warp_mode = cv2.MOTION_EUCLIDEAN
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                number_of_iterations, termination_eps)
        
    # Iterate images using first one as reference to align them
    for img in image_list[1:]:
        height, width = img.shape
        warp_matrix = None
        img_aligned = None
        best_mse = 10e6
        # Try the four reflections
        for rx,ry in [(1,1), (-1,1), (1,-1), (-1,-1)]:
            reflection = np.float32(np.asarray([[rx, 0,  width if rx == -1 else 0],
                                                [0,  ry, height if ry == -1 else 0]]))
            img_reflected = cv2.warpAffine(img, reflection, (width, height),
                                           borderValue=(255, 255, 255))
            # Try the alignment                
            try: 
                _, warp_matrix_new = cv2.findTransformECC(ref_img,
                                                          img_reflected,
                                                          warp_matrix_orig,
                                                          warp_mode,
                                                          criteria,
                                                          None,
                                                          5)
                warp_matrix_new = np.dot(np.vstack([warp_matrix_new, [0, 0, 1]]),
                                         np.vstack([reflection, [0, 0, 1]]))
                warp_matrix_new = np.float32(np.delete(warp_matrix_new, 2, 0))
                img_aligned_new = cv2.warpAffine(img, warp_matrix_new, (width, height),
                                                 borderValue=(255, 255, 255))
                # Compute alignment score to keep the best one
                new_mse = mse(ref_img, img_aligned_new)
                if new_mse < best_mse:
                    best_mse = new_mse
                    img_aligned, warp_matrix = img_aligned_new.copy(), warp_matrix_new.copy()
            except Exception as e:
                pass
        if warp_matrix is None or img_aligned is None:
            print("Images could not be aligned properly")
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            img_aligned = img
            
        # Store the aligned image and the transformation matrix
        warp_list.append(warp_matrix)
        aligned_images.append(img_aligned)
    return aligned_images, warp_list

def detect_edges(image, sigma=0.33):
    """
    Detects edges in image using
    the Canny detection algorithm and returns
    the x,y coordinates of the edges
    """
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(image, lower, upper)
    ref_indices = np.where(edges != [0])
    ref_coordinates = np.vstack((ref_indices[1], ref_indices[0])).transpose()
    return ref_coordinates

def edges_detection_aligment(image_list):
    """
    Aligns images using the first image as reference
    The methods uses edges detection and points matching to find
    the best alignment.
    Returns the aligned images and the warp transformations
    """
    ref_img = image_list[0]
    ref_coordinates = detect_edges(ref_img)
    warp_list = [np.eye(2, 3, dtype=np.float32)]
    aligned_images = [ref_img]

    # Iterate images using first one as reference to align them
    for img in image_list[1:]:
        height, width = img.shape
        warp_matrix = None
        img_aligned = None
        best_mse = 10e6
        # Try the four reflections
        for rx,ry in [(1,1), (-1,1), (1,-1), (-1,-1)]:
            reflection = np.float32(np.asarray([[rx, 0,  width if rx == -1 else 0],
                                                [0,  ry, height if ry == -1 else 0]]))
            img_reflected = cv2.warpAffine(img, reflection, (width, height),
                                           borderValue=(255, 255, 255))
            
            # Detect edges
            coordinates = detect_edges(img_reflected)
            
            # Find transformation
            min_rows = min(coordinates.shape[0], ref_coordinates.shape[0]) - 1
            warp_matrix_new, _, _ = icp(coordinates[0:min_rows,:], ref_coordinates[0:min_rows,:])
            
            # Apply transformation
            warp_matrix_new = np.dot(warp_matrix_new, np.vstack([reflection, [0, 0, 1]]))
            warp_matrix_new = np.float32(np.delete(warp_matrix_new, 2, 0))
            img_aligned_new = cv2.warpAffine(img, warp_matrix_new, (width, height),
                                             borderValue=(255, 255, 255))
            
            # Compute alignment score to keep the best one
            new_mse = mse(ref_img, img_aligned_new)
            if new_mse < best_mse:
                best_mse = new_mse
                img_aligned, warp_matrix = img_aligned_new.copy(), warp_matrix_new.copy()

        if warp_matrix is None or img_aligned is None:
            print("Images could not be aligned properly")
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            img_aligned = img
            
        # Store the aligned image and the transformation matrix
        warp_list.append(warp_matrix)
        aligned_images.append(img_aligned)
    return aligned_images, warp_list

def select_points(img):
    """
    Lets the user select points
    in an image and returns the points
    """
    posList = list()
    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img,(x,y),2,(255,0,0),-1)
            posList.append((x, y))
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', onMouse)
    while True:
        cv2.imshow('image', img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    return np.array(posList)
    
def manual_aligment(image_list):
    """
    Aligns images using the first image as reference
    The methods lets the user chooses points to be used as references
    Returns the aligned images and the warp transformations
    """
    ref_img = image_list[0]
    ref_points = select_points(ref_img.copy())
    warp_list = [np.eye(2, 3, dtype=np.float32)]
    aligned_images = [ref_img]
    
    # Iterate images using first one as reference to align them
    for img in image_list[1:]:
        height, width = img.shape
        warp_matrix = None
        img_aligned = None
        best_mse = 10e6
        
        # Get points
        points = select_points(img.copy())

        # Try the four reflections
        for rx,ry in [(1,1), (-1,1), (1,-1), (-1,-1)]:
            reflection = np.float32(np.asarray([[rx, 0,  width if rx == -1 else 0],
                                                [0,  ry, height if ry == -1 else 0]]))
            reflection = np.vstack([reflection, [0, 0, 1]])
            points_reflected = np.hstack([points, np.ones((points.shape[0], 1))])
            points_reflected = np.dot(reflection, points_reflected.transpose()).transpose()
            points_reflected = np.delete(points_reflected, 2, 1)
            
            # Find transformation
            min_rows = min(points_reflected.shape[0], ref_points.shape[0]) - 1
            warp_matrix_new, _, _ = icp(points_reflected[0:min_rows,:], ref_points[0:min_rows,:])
        
            # Apply transformation
            warp_matrix_new = np.dot(warp_matrix_new, reflection)
            warp_matrix_new = np.float32(np.delete(warp_matrix_new, 2, 0))
            img_aligned_new = cv2.warpAffine(img, warp_matrix_new, (width, height),
                                             borderValue=(255, 255, 255))
            
            # Compute alignment score to keep the best one
            new_mse = mse(ref_img, img_aligned_new)
            if new_mse < best_mse:
                best_mse = new_mse
                img_aligned, warp_matrix = img_aligned_new.copy(), warp_matrix_new.copy()
        
        if warp_matrix is None or img_aligned is None:
            print("Images could not be aligned properly")
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            img_aligned = img
            
        # Store the aligned image and the transformation matrix
        warp_list.append(warp_matrix)
        aligned_images.append(img_aligned)
        
    return aligned_images, warp_list

def project_warp_matrix(warp_list, warp_list_c, images, w, h):
    """
    Adjusts the warp transformation matrices to work on the
    original images by combining them
    """
    assert(len(warp_list) == len(images) == len(warp_list_c))
    warp_list_backtransf = list()
    for warp_matrix, warp_matrix_c, img in zip(warp_list, warp_list_c, images):
        height, width, _ = img.shape
        s_x = w / width
        s_y = h / height
        project_down = np.array([[s_x, 0, 0],
                                 [0, s_y, 0],
                                 [0, 0,  1]])
        project_up = np.linalg.inv(project_down)
        A = np.vstack([warp_matrix_c, [0, 0, 1]])
        B = np.vstack([warp_matrix, [0, 0, 1]])
        C = project_up @ B @ A @ project_down
        C = np.delete(C, 2, 0)
        warp_list_backtransf.append(C)
    return warp_list_backtransf

def transform_original_image(image_list, warp_list_backtransf):
    """
    Given a list of images and a list of affine transformation
    transform the images with the transformation and returns the
    transformed images
    """
    assert(len(warp_list_backtransf) == len(image_list))
    transformed_images = list()
    # Run trough images and apply the transformation matrix
    for img, warp_matrix in zip(image_list, warp_list_backtransf):
        height, width, _ = img.shape
        img_aligned = cv2.warpAffine(img, warp_matrix, (width, height),
                                     borderValue=(255, 255, 255))
        transformed_images.append(img_aligned)
    return transformed_images

def transform_counts(counts_list, image_list, warp_list):
    """
    Given a list of counts matrices and a list of affine transformation
    transform the spots in the transformation with the affine matrices and returns the
    transformed counts matrices
    """
    assert(len(warp_list) == len(image_list) == len(counts_list))
    transformed_counts = list()
    # Run trough counts and apply the transformation matrix
    for counts, img, warp_matrix in zip(counts_list, image_list, warp_list):
        height, width, _ = img.shape
        sx = width / 32
        sy = height / 34
        t = np.array([[sx, 0, -sx], [0, sy, -sy], [0, 0, 1]])
        t_inv = np.linalg.inv(t)
        spot_coords = np.zeros((counts.shape[0],2), dtype=np.float32)
        for i,spot in enumerate(counts.index):
            x,y = spot.split("x")
            spot_coords[i,0] = int(x)
            spot_coords[i,1] = int(y)
        spot_coords = np.hstack([spot_coords, np.ones((spot_coords.shape[0], 1))])
        M = np.vstack([warp_matrix, [0, 0, 1]])
        A = t_inv @ M @ t 
        spot_coords_adjusted = np.dot(A, spot_coords.transpose()).transpose()
        counts.index = ["{}x{}".format(abs(x), abs(y)) for x,y in zip(spot_coords_adjusted[:,0],
                                                                      spot_coords_adjusted[:,1])]
        #TODO Remove spots that are outside after alignment
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
         debug, alignment_method, normalize, border):
    
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
    
    print("Parsing images...")
    images = [cv2.cvtColor(cv2.imread(file), 
                           cv2.COLOR_BGR2RGB) for file in images_files]
       
    if border: 
        print("Adding white borders...")
        images = add_borders(images)
        if debug:
            plot_images(images, False, os.path.join("images_border.png"))
        
    print("Resizing images...")
    images_resized = [cv2.resize(img, (down_width, down_height), 
                                 interpolation=cv2.INTER_CUBIC) for img in images]
    if debug:
        plot_images(images_resized, False, os.path.join("images_resized.png"))
        
    print("Computing masks for the images...")
    binary_masks = get_binary_masks(images_resized)
    if debug:
        plot_images(binary_masks, True, os.path.join(outdir, "masks.png"))
    masked_images = mask_images(images_resized, binary_masks)
    if debug:
        plot_images(masked_images, False, os.path.join(outdir, "masked_images.png"))
    
    print("Centering images...")
    masked_images, binary_masks, warp_list_center = center_images(masked_images, binary_masks)
    if debug:
        plot_images(masked_images, False, os.path.join(outdir, "centered_images.png"))
        
    print("Performing alignment...")
    grey_masked_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in masked_images]
    if normalize:
        grey_masked_images = [cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX) for img in grey_masked_images]
        grey_masked_images = [grey_masked_images[0]] + \
                             [hist_norm(img, grey_masked_images[0]) for img in grey_masked_images[1:]]
    if debug:
        plot_images(grey_masked_images, True, os.path.join("grey_images.png"))
        
    if alignment_method == "EDGES":
        aligned_images, warp_list = edges_detection_aligment(binary_masks)
    elif alignment_method == "MANUAL":
        aligned_images, warp_list = manual_aligment(grey_masked_images)
    else:
        aligned_images, warp_list = euclidean_alignment(grey_masked_images)
    if debug:
        plot_images(aligned_images, True, os.path.join("aligned_grey_images.png"))
    
    print("Transforming original images...")
    warp_list_backtransf = project_warp_matrix(warp_list, warp_list_center, images, down_width, down_height)
    transformed_original_images = transform_original_image(images, warp_list_backtransf)
    if debug:
        plot_images(transformed_original_images, False, os.path.join("aligned_images.png"))
            
    if border:
        print("Removing borders...")
        transformed_original_images = remove_borders(transformed_original_images)
     
    print("Transforming original counts matrices...")
    transformed_counts = transform_counts(counts, transformed_original_images, warp_list_backtransf)
    
    # Save transformations
    for tr, name in zip(warp_list_backtransf, counts_files):
        clean_name = os.path.basename(name).split(".")[0]
        np.savetxt(os.path.join(outdir, "aligment_{}.txt".format(clean_name)), tr, delimiter="\t")
        
    # Save images
    for img, name in zip(transformed_original_images, images_files):
        clean_name = os.path.basename(name).split(".")[0]
        matplotlib.image.imsave(os.path.join(outdir, "aligned_{}.jpg".format(clean_name)), img)
    
    # Save counts
    for counts, name in zip(transformed_counts, counts_files):
        clean_name = os.path.basename(name).split(".")[0]
        counts.to_csv(os.path.join(outdir, "aligned_{}.tsv".format(clean_name)), sep="\t")
        
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
    parser.add_argument("--border", action="store_true", default=False,
                        help="Whether to add a white border to images prior alignment (recommended for EDGES alignment method)")
    parser.add_argument("--normalize", action="store_true", default=False,
                        help="Whether to perform a cumulative histogram normalization on the images\n" \
                        "to equalize the intensities (recommended for ECC alignment method")
    parser.add_argument("--alignment-method", default="ECC", metavar="[STR]", 
                        type=str, 
                        choices=["ECC", "EDGES", "MANUAL"],
                        help="The method to use for the image alignment:\n" \
                        "ECC = automatic alignment based on the brightness and contour (ECC algorithm) \n" \
                        "EDGES = automatic alignment based on edges detection\n" \
                        "MANUAL = manual alignment based selected points\n" \
                        "(default: %(default)s)")
    args = parser.parse_args()
    main(args.counts, args.images, args.down_width, args.down_height, args.outdir,
         args.debug, args.alignment_method, args.normalize, args.border)