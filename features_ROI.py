'''
Subsetted features with higher likelihood of detecting spiculations
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

def convert_scale_alpha(maxval):
    '''
    Convert Image Scale
    '''
    return 255.0/maxval

def plot_comparison(original, filtered):
    '''
    Compare plots juxtaposed
    '''
    fig,axes = plt.subplots(1, 2)
    fig.set_size_inches([12, 9])
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('original')
    axes[1].imshow(filtered, cmap='gray')
    axes[1].set_title('filtered')


def generate_edges(original):
    '''
    Ensemble of binary and canny method to detect clean spiculation border
    to do: add area maybe
    returns edges
    '''

    thresh = threshold_mean(original) # binarise image
    binary = original > thresh
    edges = canny(binary, sigma=5)

    return edges



def generate_hog(original):
    '''
    Histogram of Oriented Gradient: feature descriptor for object detection
    returns generated feature and recaled image for comparison
    '''
    fd, hog_image = hog(original, orientations=5, pixels_per_cell=(5, 5),
                    cells_per_block=(2, 1), visualize=True, multichannel=False)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 255)) # Enhance constrast
    return fd, hog_image_rescaled

def generate_snake(original):

    '''
    Active contour model, modified region
    returns feature and initialising location
    ''' 

    spic_area = 0.01*original.size
    s = np.linspace(0, 2*np.pi, spic_area)
    r = 200 + 100*np.sin(s)
    c = 220 + 100*np.cos(s)
    init = np.array([r, c]).T
    snake = active_contour(gaussian(original, 3),
                             init, alpha=0.015, beta=10, gamma=0.001)
    return snake, init

def generate_gabor(image):
    '''
    Gabor features
    '''    
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
        filtered = ndimage.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats

def gabor_match(feats, ref_feats):
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i



