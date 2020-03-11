#============================================================================#
# IMAGE PRE-PROCESSING
#============================================================================#
import numpy as np
import os, argparse 
import pydicom, cv2
import scipy
import matplotlib.pyplot as plt
from skimage import exposure, img_as_float
from skimage.segmentation import (morphological_chan_vese,
                                  checkerboard_level_set)
from skimage.filters import threshold_otsu
from skimage.morphology import opening
from skimage.measure import label
from sklearn.utils.testing import ignore_warnings


@ignore_warnings(category=FutureWarning)
def threshold_img(img, pctile=50):
    '''
    Rescales image intensity and thresholds
    Takes: image pixel array, percentile threshold value
    Returns: thresholded binary image pixel array
    '''

    # rescaling --> consider exposure.equalize_hist
    p_thresh, p100 = np.percentile(img, (pctile, 100))
    img_scaled = exposure.rescale_intensity(img, in_range=(p_thresh, p100))

    # thresholding
    img_thresh = threshold_otsu(img_scaled)

    # binarize
    return img_scaled > img_thresh


def store_evolution_in(lst):
    '''
    Returns a callback function to store the evolution of the level sets in
    the given list.
    '''

    def _store(x):
        lst.append(np.copy(x))

    return _store


def apply_ACWE(img):
    '''
    Segments largest region using ACWE.
    '''
    image = img_as_float(img)
    init_ls = checkerboard_level_set(image.shape, 6)
    evolution = []
    callback = store_evolution_in(evolution)
    ls = morphological_chan_vese(image, 3, init_level_set=init_ls,
                                 smoothing=1, iter_callback=callback)
    #plt.figure(figsize=(9, 3))
    #plt.imshow(ls, cmap="gray")
    #plt.contour(evolution[3], [0.5], colors='y')

    return (1 - ls)


# label image regions
def define_region(img):
    '''
    Keeps the largest region only.
    '''
    labels = label(img)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=img.flat))
    main = (1 * largestCC)
    
    return main


def fill_holes(img):
    '''
    Fills all holes.
    '''
    filled = scipy.ndimage.morphology.binary_fill_holes(img)

    return (1 * filled)


def go(file):
    '''
    Run all functions.
    '''
    original = pydicom.dcmread(file)
    binary = threshold_img(original.pixel_array, pctile=50)
    segment = apply_ACWE(binary)
    main_region = define_region(segment)
    filled = fill_holes(main_region)

    return filled, original



if __name__ == "__main__":

    benign_path = "raw/Mass-Training_P_00187_LEFT_CC_1-07-21-2016-DDSM-85364-1-ROI_mask_images-25005-000000.dcm"
    malignant_path = "raw/Mass-Training_P_00149_LEFT_CC_1-07-21-2016-DDSM-06526-1-ROI_mask_images-57657-000001.dcm"

    # benign, orig_benign = go(benign_path)
    # malignant, orig_malignant = go(malignant_path)