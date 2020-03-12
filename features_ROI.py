'''
Subsetted features with higher likelihood of detecting spiculations
Functions starting with generate produce single valued integer features 
'''
import os
import argparse
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
# import preprocess as p

# from skimage.measure import label, regionprops, regionprops_table
from skimage.color import label2rgb
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import minmax_scale
from skimage.feature import hog
from skimage import data, exposure
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.filters import threshold_mean
from skimage.feature import canny
from scipy import ndimage
from skimage.transform import hough_line, hough_line_peaks


def helper_convert_scale_alpha(maxval):
    '''
    Helper function to convert Image Scale
    Returns: rescaled array
    '''
    return 255.0/maxval

def helper_plot_comparison(orig, filtered):
    '''
    Compare plots juxtaposed
    '''
    fig,axes = plt.subplots(1, 2)
    fig.set_size_inches([12, 9])
    axes[0].imshow(orig, cmap='gray')
    axes[0].set_title('original')
    axes[1].imshow(filtered, cmap='gray')
    axes[1].set_title('filtered')

def helper_bbox(img):
    '''
    Returns: Bounding Box over ROI and Segmented region
    used in subsequent function get_iou

    '''

    a = np.where(img != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox

def generate_entropy(original):
    '''
    function returns entropy of a signal
    signal must be a 1-D numpy array
    Returns: entropy measure
    '''
    orig = original.pixel_array
    
    lensig=signal.size
    symset=list(set(signal))
    numsym=len(symset)
    propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]
    ent=np.sum([p*np.log2(1.0/p) for p in propab])
    return ent

def generate_iou(original, segmented_mask):
    '''
    Get intersection area using bounding box defined on area
    Note: The dimensions of superset (A) is considered as those of main ROI
    PS: DICE did not have much variance 
    Returns: intersected area
    '''
    orig = original.pixel_array
    boxA = np.array(helper_bbox(orig))
    boxB = np.array(helper_bbox(segmented_mask))

    xA = max(boxA[0], boxB[0]) + 1
    yA = max(boxA[1], boxB[1]) + 1
    xB = min(boxA[2], boxB[2]) + 1
    yB = min(boxA[3], boxB[3]) + 1

    # compute the area of intersection rectangle
    # interArea = max(0, abs(xB - xA) + 1) * max(0, abs(yB - yA) + 1) 
    interArea = abs((xB - xA) * (yB - yA)) # improved measure

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = abs((boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1))
    boxBArea = abs((boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = abs(interArea / float(boxAArea + boxBArea - interArea))

    return iou


def helper_edges(original):
    '''
    Ensemble of binary and canny method to detect clean spiculation border
    to do: used in hough
    Returns edges
    '''
    orig = original.pixel_array

    thresh = threshold_mean(orig) # binarise image
    binary = orig > thresh
    edges = canny(binary, sigma=5)

    return edges

def generate_hough(original):
    '''
    Active contour model
        Define segmented mask proportional to the target mass region 
    Returns number of lines
    ''' 
    orig = original.pixel_array
    lines = []
    edges = helper_edges(original)  # get edges from canny
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 100) # permute over angles
    h, theta, d = hough_line(edges, theta=tested_angles)


    origin = np.array((0, orig.shape[1]))
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        inter_lines = abs(y0-y1) 
        lines.append(inter_lines)
    num_lines = len(lines)
    return num_lines


def generate_snake(original):
    '''
    Active contour model
        Define segmented mask proportional to the target mass region 
    returns rugged mean dist from defined centroid at r,c
    ''' 
    orig = original.pixel_array

    spic_area = round(0.01*orig.size)
    s = np.linspace(0, 2*np.pi, spic_area)
    r = 200 + 100*np.sin(s)
    c = 220 + 100*np.cos(s)
    init = np.array([r, c]).T
    snake = active_contour(gaussian(orig, 3),
                             init, alpha=0.015, beta=10, gamma=0.001)

    snake_array = np.asarray(snake)
    dist = np.sqrt((r-snake_array[:, 0])**2 +(c-snake_array[:, 1])**2)
    mean_dist = int(np.mean(dist))
    return mean_dist


def generate_gabor(original):
    '''
    Gabor features
    '''    
    orig = original.pixel_array
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)

    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndimage.convolve(orig, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return len(feats)
