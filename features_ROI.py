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

pd.set_option('display.max_columns', 500)



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

def sk_hog(original, filtered):
    '''
    Histogram of Oriented Gradient: feature descriptor for object detection
    returns generated feature
    '''
    fd, hog_image = hog(ms, orientations=5, pixels_per_cell=(5, 5),
                    cells_per_block=(2, 1), visualize=True, multichannel=False)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 255)) # Enhance constrast
    plot_comparison(ms, hog_image_rescaled)
    return fd

def morph_snake(original, filtered):
  '''
  ''' 
  s = np.linspace(0, 2*np.pi, 400)
  r = 100 + 100*np.sin(s)
  c = 220 + 100*np.cos(s)
  init = np.array([r, c]).T
  snake = active_contour(gaussian(ms, 3),
                         init, alpha=0.015, beta=10, gamma=0.001)
  return snake

def gabor_feat(image, kernels):
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
        filtered = nd.convolve(image, kernel, mode='wrap')
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



