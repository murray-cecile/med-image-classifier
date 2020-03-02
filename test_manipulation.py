#==============================================================================#
# TEST IMAGE MANIPULATION
#
#==============================================================================#

import numpy as np
import os, pydicom, cv2
import matplotlib.pyplot as plt

from skimage import exposure
from skimage.filters import gaussian
from skimage.filters import try_all_threshold
from skimage.segmentation import active_contour


import data_download as dd


TEST_FILE = "raw/Mass-Training_P_00004_RIGHT_MLO_1-07-21-2016-DDSM-83774-1-ROI_mask_images-84846-000000.dcm"
TEST_MASK = "raw/Mass-Training_P_00004_RIGHT_MLO_1-07-21-2016-DDSM-83774-1-ROI_mask_images-84846-000001.dcm"
TEST_FULL = "raw/Mass-Training_P_00004_RIGHT_MLO-07-20-2016-DDSM-24486-1-full_mammogram_images-89890-000000.dcm"


def get_dicom_pixel_array(file):
    '''
    Extract the pixel array from a pydicom file and convert to 8 bit
    Takes: string filepath
    Returns: pixel array
    '''
    a = dd.open_img(file).pixel_array
    return (a/256).astype('uint8') 



if __name__ == "__main__":
    
    ds = dd.open_img(TEST_FILE)   
    a8 = get_dicom_pixel_array(TEST_FILE)
    plt.imshow(a8)
    plt.show()

    m = dd.open_img(TEST_MASK)
    full = dd.open_img(TEST_FULL)

    plt.imshow(m.pixel_array)
    plt.show()
    plt.imshow(full.pixel_array)
    plt.show()
        
    # this performs terribly
    # b = cv2.adaptiveThreshold(a8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # plt.imshow(b)

    # try different histogram equalization methods (skimage)
    a8_eq = exposure.equalize_hist(a8)

    p90, p100 = np.percentile(a8, (90, 100))
    a8_ct = exposure.rescale_intensity(a8, in_range=(p90, p100))


    # fig, ax = try_all_threshold(a8_ct, figsize=(10, 8), verbose=False)
    # plt.show()

    # try more binary
    # ret, b = cv2.threshold(a8_eq, 230, 255, cv2.THRESH_BINARY)
    # plt.imshow(b)
    # plt.show()    

    # # active contour from skimage
    # s = np.linspace(0, 2*np.pi, 400)
    # r = 100 + 100*np.sin(s)
    # c = 220 + 100*np.cos(s)
    # init = np.array([r, c]).T
    # snake = active_contour(gaussian(a8, 3),
    #                    init, alpha=0.015, beta=10, gamma=0.001,
    #                    coordinates='rc')
    # fig, ax = plt.subplots(figsize=(7, 7))
    # ax.imshow(a8, cmap=plt.cm.gray)
    # ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
    # ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
    # ax.set_xticks([]), ax.set_yticks([])
    # ax.axis([0, a8.shape[1], a8.shape[0], 0])
    # plt.show()