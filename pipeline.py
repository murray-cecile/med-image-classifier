'''
Full pipeline
'''
import numpy as np
import pydicom
import matplotlib.pyplot as plt

from skimage import exposure, img_as_float
from skimage.segmentation import morphological_chan_vese
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops, regionprops_table
from skimage.color import label2rgb
import scipy


FILE = r'./Mass-Training_P_00059_LEFT_CC_1-07-20-2016-DDSM-17695-1-cropped_images-02767-000000.dcm'
MASK = r'./Mass-Training_P_00059_LEFT_CC_1-07-21-2016-DDSM-38707-1-ROI_mask_images-82600-000000.dcm'


def open_img(path):
    '''
    '''
    ds = pydicom.dcmread(path)

    return ds


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
    binary = img_scaled > img_thresh
    
    return binary


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
    Segments largest region using ACWE
    '''
    image = img_as_float(img)
    init_ls = checkerboard_level_set(image.shape, 6)
    evolution = []
    callback = store_evolution_in(evolution)
    ls = morphological_chan_vese(image, 3, init_level_set=init_ls, smoothing=1, iter_callback=callback)
    #plt.figure(figsize=(9, 3))
    #plt.imshow(ls, cmap="gray")
    #plt.contour(evolution[3], [0.5], colors='y')

    return (1 - ls)


# label image regions
def define_region(img):
    '''
    Keeps the largest region only
    '''
    labels = label(img)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=img.flat))
    main = (1 * largestCC)
    
    return main


def fill_holes(img):
    '''
    '''
    filled = scipy.ndimage.morphology.binary_fill_holes(img)

    return (1 * filled)


def properties(img):
    '''
    '''
    label_image = measure.label(img)
    props = regionprops_table(label_image, img,
                              properties=['area', 'centroid', 'coords'])


def go():
    '''
    '''
    img = open_img(FILE)
    binary = threshold_img(img.pixel_array, pctile=50)
    ls = apply_ACWE(binary)
    main_region = define_region(ls)
    filled = fill_holes(main_region)

    return filled










