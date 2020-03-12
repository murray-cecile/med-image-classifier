#============================================================================#
# IMAGE PRE-PROCESSING
#============================================================================#
import numpy as np
import os, argparse 
import pydicom, cv2
import scipy
from statistics import mode
import matplotlib.pyplot as plt
from skimage import exposure, img_as_float
from skimage.segmentation import (morphological_chan_vese,
                                  checkerboard_level_set)
from skimage.filters import threshold_otsu
from skimage.morphology import opening
from skimage.measure import label
from sklearn.utils.testing import ignore_warnings

# DELETE THIS LATER 
from skimage.measure import label, regionprops, regionprops_table



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

    # Figure out what background is, make sure it's 0
    original = img_scaled > img_thresh
    original = img_as_float(original)

    return original


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
    init_ls = checkerboard_level_set(img.shape, 6)
    evolution = []
    callback = store_evolution_in(evolution)
    l = morphological_chan_vese(img, 3, init_level_set=init_ls,
                                smoothing=1,
                                iter_callback=callback)
    
    #plt.figure(figsize=(9, 3))
    #plt.imshow(ls, cmap="gray")
    #plt.contour(evolution[3], [0.5], colors='y')

    return l


def check_segmentation(a):
    '''
    Confirm lesion is labeled with 1's by segmentation by checking 
        % of edge pixels that are 1 and reversing 0/1's if > 50%
    Takes: segmented image array
    Returns: segmented image array with "correct" 0/1 assignment
    '''

    x, y = a.shape

    t = a[0,:].sum()
    l = a[:,0].sum()
    r = a[:,y-1].sum()
    b = a[x-1,:].sum()
    perimeter = 2 * (x + y)

    if (t + l + r + b) / perimeter > 0.5:
        return np.where(a == 0, 1, 0)
    else:
        return a


# label image regions
def define_region(img):
    '''
    Keeps the largest region only.
    '''
    # Figure out what the background is
    # Whatever the majority of the four corners is = background
    labels = label(img)
    largestCC = labels == np.argmax(np.bincount(labels.flat,
                                                weights=img.flat))
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
    original = pydicom.dcmread(file, force=True)
    binary = threshold_img(original.pixel_array, pctile=50)
    segment = apply_ACWE(binary)
    main_region = define_region(segment)
    filled = fill_holes(main_region)
    confirmed_filled = check_segmentation(filled)

    return confirmed_filled, original


if __name__ == "__main__":

    benign_path = "raw_train/Mass-Training_P_00187_LEFT_CC_1-07-21-2016-DDSM-85364-1-ROI_mask_images-25005-000000.dcm"
    malignant_path = "raw_train/Mass-Training_P_00149_LEFT_CC_1-07-21-2016-DDSM-06526-1-ROI_mask_images-57657-000001.dcm"

    # benign, orig_benign = go(benign_path)
    # malignant, orig_malignant = go(malignant_path)


    
    list_of_files = os.listdir("raw_train/")
    for file in list_of_files[0:2]:
        try:
            full_path = "raw_train/" + file
            original = pydicom.dcmread(full_path, force=True)
            binary = threshold_img(original.pixel_array, pctile=50)
            segment = apply_ACWE(binary)
            main_region = define_region(segment)
            filled = fill_holes(main_region)
            confirmed_filled = check_segmentation(filled)


            label_image = label(confirmed_filled)
            
            
            fig, axes = plt.subplots(2, 2, figsize = (8, 8))
            ax = axes.ravel()
            ax[0].imshow(original.pixel_array, cmap="gray")
            ax[1].imshow(segment, cmap="gray")
            ax[2].imshow(main_region, cmap="gray")
            ax[3].imshow(confirmed_filled, cmap="gray")
            fig.tight_layout()
            plt.show()
        except Exception as e:
            print('Could not process: ', file)
            print(e)
