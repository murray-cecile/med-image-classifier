#==============================================================================#
# TEST IMAGE MANIPULATION
#
# Cecile Murray
#==============================================================================#

import numpy as np
import os, pydicom, cv2
import matplotlib.pyplot as plt

from skimage import exposure
from skimage.filters import gaussian
from skimage.segmentation import active_contour

import data_download as dd

TEST_FILE = "raw/CBIS-DDSM-Calc-Test_P_00331_LEFT_CC_1-08-29-2017-DDSM-53305-1-ROI_mask_images-07190-000000.dcm"
TEST_MASK = "raw/CBIS-DDSM-Calc-Test_P_00331_LEFT_CC_1-08-29-2017-DDSM-53305-1-ROI_mask_images-07190-000001.dcm"

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

    ds2 = dd.open_img(TEST_MASK)
    m8 = get_dicom_pixel_array(TEST_MASK)
        
    # this performs terribly
    # b = cv2.adaptiveThreshold(a8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # plt.imshow(b)

    # try more binary
    ret, b = cv2.threshold(a8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    plt.imshow(b)
    plt.show()    

    # also don't get anything out of this
    # c = cv2.Canny(a8, 125, 255)
    
    # # try histogram equalization (skimage)
    # a8_eq = exposure.equalize_hist(a8)

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