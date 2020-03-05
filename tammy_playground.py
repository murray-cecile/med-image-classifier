'''
Tammy's Steps:

Step 1: Data available here: https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM#5e40bd1f79d64f04b40cac57ceca9272
Step 2: Download 'Mass-Training ROI and Cropped Images'. Move these to the raw folder in med-image-classifier.
Step 3: Use NBIA data retriever to convert images from .tcia to .dcm (can pause at any time)
Step 4: Flatten downloaded folders using the following linux command:
        bash ingest-data.sh /Users/Tammy/Documents/_MSCAPP/Winter_2020/Computer_Vision_MP/med-image-classifier/raw/
Step 5: Remove the filepath portion of directory name from all files using the following Linux command:
        rename -- 's/-Users-Tammy-Documents-_MSCAPP-Winter_2020-Computer_Vision_MP-med-image-classifier-raw--CBIS-DDSM-//' *
Step 6: Delete the masks using the following Linux command, leaving only the ROIs:
        find . -name '*-000001.dcm' -delete
'''
import numpy as np
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
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
# Feature 1: 


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
    plt.figure(figsize=(9, 3))
    plt.imshow(ls, cmap="gray")
    plt.contour(evolution[3], [0.5], colors='y')


# label image regions
def define_region(img):
    '''
    Keeps the largest region only
    '''
    labels = label(img)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=img.flat))
    return largestCC


