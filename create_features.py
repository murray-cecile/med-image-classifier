'''
Playground
'''
import numpy as np
import pandas as pd
import os, pydicom, cv2
import matplotlib.pyplot as plt

from scipy import ndimage
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

def compute_scharr(img, mask = None):
    '''
    Compute the gradient direction and magnitude using Scharr filter 
    Takes: segmented image array
    Returns: array of angles
    '''

    s_h = scharr_h(img, mask = mask)
    s_v = scharr_v(img, mask = mask)
    return np.arctan2(s_v, s_h), np.sqrt(s_v**2 + s_h**2)


def compute_gradient_std(theta, magnitude):
    '''
    Takes: array of angles and array of magnitudes from Sobel/Scharr
    Returns: standard deviation of normalized magnitude-angle distribution
    '''

    avg = np.mean(magnitude)
    # print("average magnitude is ", avg)
    x_dim, y_dim = theta.shape

    t = theta.reshape((x_dim * y_dim, 1))
    m = magnitude.reshape((x_dim * y_dim, 1))
    df = pd.DataFrame(np.hstack((t, m)), columns = ["theta", "magnitude"])

    if avg != 0:
        edge_gradient_dist = df.groupby(['theta'], as_index=False).sum() / avg
        return edge_gradient_dist['magnitude'].std()
    
    else: 
        return 0


def get_border_pixels(mask):
    '''
    Extract the pixels on the border of a region
    Takes: segmented mask
    Returns: array mask where border pixels are 1, all others 0
    '''

    distance = ndimage.distance_transform_edt(mask)
    distance[distance != 1] = 0
    return distance


# TO DO: try more of the neighborhoods in Huo and Giger (1995)
def compute_spiculation(original, segmented_mask):

    orig = original.pixel_array

    # filter approach based on Huo and Giger (1995)
    # they use Sobel but Scharr is supposed to be rotation invariant (?)
    theta_A, magnitude_A = compute_scharr(orig, mask = segmented_mask)
    std_dev_A = compute_gradient_std(theta_A, magnitude_A)

    # B: just use the 1 pixel border
    border_mask = get_border_pixels(segmented_mask)
    theta_B, magnitude_B = compute_scharr(orig, mask = border_mask)
    std_dev_B = compute_gradient_std(theta_B, magnitude_B)

    # TO DO: use 20 pixels inside and outside border

    # TO DO: repeat the above but using an opening filter (20% of size, circular)

    # higher standard deviation here indicates more spiculation
    return {'A': std_dev_A, 'B': std_dev_B}
 

def make_all_features(original, filled):
    '''
    Runs all manual feature generation features on image
    Takes: original image (dicom) and region mask pixel array
    Returns: single row of data frame with features computed
    '''
    
    spiculation = compute_spiculation(original, filled)
    mf = {'spiculationA': spiculation['A'], \
        'spiculationB': spiculation['B']}
    
    return mf


if __name__ == "__main__":
    
    benign_path = "raw/Mass-Training_P_00169_RIGHT_MLO_1-07-21-2016-DDSM-75457-1-ROI_mask_images-57822-000001.dcm"
    malignant_path = "raw/Mass-Training_P_00149_LEFT_CC_1-07-21-2016-DDSM-06526-1-ROI_mask_images-57657-000001.dcm"

    # read in a test image to play with
    benign, orig_benign = p.go(benign_path)
    malignant, orig_malignant = p.go(malignant_path)

    # plt.imshow(orig_benign.pixel_array)
    # plt.show()
    # plt.imshow(orig_malignant.pixel_array)
    # plt.show()

    # b_thetas, b_magnitudes = compute_sobel(orig_benign.pixel_array, benign)
    # m_thetas, m_magnitudes = compute_sobel(orig_malignant.pixel_array, malignant)

    # print(compute_spiculation(orig_benign, benign))
    # print(compute_spiculation(orig_malignant, malignant))

    # make_all_features(orig_malignant, malignant)

    # I don't know how to interpret the results of this
    # hough_trans = hough_line(filled_img)













