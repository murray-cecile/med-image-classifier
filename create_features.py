'''
Playground
'''
import numpy as np
import pandas as pd
import os, pydicom, cv2
import matplotlib.pyplot as plt

from skimage import exposure
from skimage.filters import gaussian
from skimage.filters import try_all_threshold
from skimage import data, img_as_float
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set,
                                  chan_vese,
                                  active_contour)
from skimage.util.shape import view_as_blocks                                  
from skimage.filters import threshold_otsu
from skimage.filters import sobel
from skimage.filters import sobel_h
from skimage.filters import sobel_v
from skimage.filters import scharr_h
from skimage.filters import scharr_v
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage.transform import hough_line

import preprocess as p

pd.set_option('display.max_columns', 500)

def apply_contour(img):
    '''
    Active contour from skimage
    '''

    s = np.linspace(0, 2*np.pi, 400)
    r = 100 + 100*np.sin(s)
    c = 220 + 100*np.cos(s)
    init = np.array([r, c]).T
    snake = active_contour(gaussian(img, 3),
                       init, alpha=0.015, beta=10, gamma=0.001,
                       coordinates='rc')
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
    ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, a8.shape[1], a8.shape[0], 0])
    plt.show()

def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store

def apply_ACWE(img):
    '''
    '''
    image = img_as_float(img)
    init_ls = checkerboard_level_set(image.shape, 6)
    # List with intermediate results for plotting the evolution
    evolution = []
    callback = store_evolution_in(evolution)
    ls = morphological_chan_vese(image, 35, init_level_set=init_ls, smoothing=3,
                                 iter_callback=callback)
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    ax = axes.flatten()
    ax[1].imshow(ls, cmap="gray")
    ax[1].set_axis_off()
    contour = ax[1].contour(evolution[2], [0.5], colors='g')
    contour.collections[0].set_label("Iteration 2")
    contour = ax[1].contour(evolution[7], [0.5], colors='y')
    contour.collections[0].set_label("Iteration 7")
    contour = ax[1].contour(evolution[-1], [0.5], colors='r')
    contour.collections[0].set_label("Iteration 35")
    ax[1].legend(loc="upper right")
    title = "Morphological ACWE evolution"
    ax[1].set_title(title, fontsize=12)


# label image regions
def define_region(img):
    '''
    '''
    labels = label(img)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=img.flat))
    return largestCC

def compute_sobel(img, mask = None):
    '''
    Compute the gradient direction and magnitude using Sobel filter 
    Takes: image pixel array, optional mask
    Returns: array of angles
    '''

    s_h = sobel_h(img, mask = mask)
    s_v = sobel_v(img, mask = mask)
    return np.arctan2(s_v, s_h), np.sqrt(s_v**2 + s_h**2)

def compute_scharr(img):
    '''
    Compute the gradient direction and magnitude using Scharr filter 
    Takes: segmented image array
    Returns: array of angles
    '''

    s_h = sobel_h(img)
    s_v = sobel_v(img)
    return np.arctan2(s_v, s_h), np.sqrt(s_v**2 + s_h**2)

# THIS IS REALLY SLOW AND POSSIBLY INCORRECT
def compute_local_std(theta, magnitude):
    '''
    Compute standard deviation of radial angle/magnitude distribution for 5x5 
        pixel neighborhoods
    Takes: arrays of thetas and gradient magnitudes from Sobel or Scharr
    Returns: array of standard deviations from each 5x5 pixel neighborhood
    '''

    x_dim = theta.shape[0] - theta.shape[0] % 5
    y_dim = theta.shape[1] - theta.shape[1] % 5

    # create 5x5 chunks (ignoring max of 4 pixels on right and bottom edge)
    theta_nhoods = view_as_blocks(theta[:x_dim, :y_dim], block_shape = (5, 5))
    magnitude_nhoods = view_as_blocks(magnitude[:x_dim, :y_dim], block_shape = (5, 5))
    
    # for each neighborhood:
        # sum the gradient magnitudes by radial angle
        # divide by average gradient magnitude
        # compute 2 * standard deviation to get "full width at half max"
    a_shape = theta_nhoods.shape
    fwhm = np.zeros(a_shape[0] * a_shape[1]).reshape(a_shape[0], a_shape[1])

    for i in range(0, a_shape[0]):
        for j in range(0, a_shape[1]):

            t = theta_nhoods[i, j].reshape([25, 1])
            m = magnitude_nhoods[i, j].reshape([25, 1])
            df = pd.DataFrame(np.hstack((t, m)), columns = ["theta", "magnitude"])
            df_mean = df['magnitude'].mean() 

            if df_mean != 0:
                edge_gradient_dist = df.groupby(['theta'], as_index=False).sum() / df_mean
                # edge_gradient_dist.reset_index                
                fwhm[i, j] = 2 * np.std(edge_gradient_dist['theta'])

    return fwhm

# INCOMPLETE
def compute_spiculation(original, segmented_mask):

    # Sobel filter approach based on Huo and Giger (1995)
    theta, magnitude = compute_sobel(original.pixel_array, mask = segmented_mask)

    # arrays with more positive values here should be more spiculated
    out_array = compute_local_std(theta, magnitude)

    return out_array


def go():
    '''
    '''
    pass



if __name__ == "__main__":
    
    benign_path = "raw/Mass-Training_P_00094_RIGHT_CC_1-07-21-2016-DDSM-28205-1-ROI_mask_images-66357-000001.dcm"
    malignant_path = "raw/Mass-Training_P_00068_RIGHT_CC_1-07-21-2016-DDSM-82707-1-ROI_mask_images-31039-000001.dcm"

    # read in a test image to play with
    benign, orig_benign = p.go(benign_path)
    malignant, orig_malignant = p.go(malignant_path)

    # plt.imshow(orig_benign.pixel_array)
    # plt.show()
    # plt.imshow(orig_malignant.pixel_array)
    # plt.show()


    # I don't know how to interpret the results of this
    # hough_trans = hough_line(filled_img)













