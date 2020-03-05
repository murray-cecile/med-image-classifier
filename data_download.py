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
import png, os, pydicom
import matplotlib.pyplot as plt
import rasterio

PATH = r'/Users/Tammy/Documents/_MSCAPP/Winter_2020/Computer_Vision_MP/med-image-classifier/raw_train'
FILE = r'Mass-Training_P_00059_LEFT_CC_1-07-20-2016-DDSM-17695-1-cropped_images-02767-000000.dcm'

# Run this in directory of interest
# Goal: create a json containing each file as well as characteristics
def open_img(single_file_path):
    '''
    '''
    ds = pydicom.dcmread(single_file_path)

    return ds


def plt_img(pydicom_img):
    '''
    '''
    plt.imshow(pydicom_img)

'''
# Not necessary
# Run this while in your folder of files
def convert_dicom_png(path=PATH):
    '''
    '''
    list_of_files = os.listdir(path)
    for file in list_of_files:
        try:
            # Note we are working with 16 bit data so values from 0 to 65535
            ds = pydicom.dcmread(file)
            shape = ds.pixel_array.shape

            # Convert to float to avoid overflow or underflow losses.
            image_2d = ds.pixel_array.astype(float)

            # Rescaling grey scale between 0-255
            image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

            # Convert to uint
            image_2d_scaled = np.uint8(image_2d_scaled)

            # Write the PNG file
            with open(destination, 'wb') as png_file:
                w = png.Writer(shape[1], shape[0], greyscale=True)
                w.write(png_file, image_2d_scaled)
        except:
            print('Could not convert: ', file)
'''