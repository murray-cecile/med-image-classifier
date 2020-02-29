#==============================================================================#
# TEST IMAGE MANIPULATION
#
# Cecile Murray
#==============================================================================#

import numpy as np
import os, pydicom, cv2
import matplotlib.pyplot as plt

import data_download as dd

TEST_FILE = "raw/CBIS-DDSM-Calc-Test_P_00331_LEFT_CC_1-08-29-2017-DDSM-53305-1-ROI_mask_images-07190-000000.dcm"

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
        
    # this performs terribly
    # b = cv2.adaptiveThreshold(a8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # plt.imshow(b)

    c = cv2.Canny(a8, 0, 255)