#==============================================================================#
# IMAGE THRESHOLDING AND PRE-PROCESSING
#
#==============================================================================#

import numpy as np
import os, argparse 
import pydicom, cv2
import matplotlib.pyplot as plt

from skimage import exposure
from skimage.filters import threshold_otsu
from skimage.morphology import opening

import data_download as dd

TEST_FILE = "raw/Mass-Training_P_00004_RIGHT_MLO_1-07-21-2016-DDSM-83774-1-ROI_mask_images-84846-000000.dcm"
TEST_MASK = "raw/Mass-Training_P_00004_RIGHT_MLO_1-07-21-2016-DDSM-83774-1-ROI_mask_images-84846-000001.dcm"
TEST_FULL = "raw/Mass-Training_P_00004_RIGHT_MLO-07-20-2016-DDSM-24486-1-full_mammogram_images-89890-000000.dcm"



def threshold_img(img, pctile):
    '''
    Rescales image intensity and thresholds
    Takes: image pixel array, percentile threshold value
    Returns: thresholded binary image pixel array
    '''

    # rescaling
    p_thresh, p100 = np.percentile(img, (pctile, 100))
    img_scaled = exposure.rescale_intensity(img, in_range=(p_thresh, p100))

    # thresholding
    img_thresh = threshold_otsu(img_scaled)

    # binarize
    return img_scaled > img_thresh


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-path", "--path", help = "Raw image file path")
    args = parser.parse_args()

    for file in os.listdir(args.path):

        try:
            file_path = args.path + '/' + file
            a = dd.open_img(file_path).pixel_array

            fig, axes = plt.subplots(1, 2, figsize=(8, 8))
            ax = axes.flatten()

            ax[0].imshow(a)

            b = threshold_img(a, 90)
            ax[1].imshow(b)
            
            plt.show()
            
        except:
            print('Could not convert: ', file)