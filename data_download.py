import numpy as np
import png, os, pydicom
import matplotlib.pyplot as plt
import rasterio

PATH = r'/Users/Tammy/Documents/_MSCAPP/Winter_2020/Computer_Vision_MP/med-image-classifier/raw_train'
FILE = r'./Mass-Training_P_00059_LEFT_CC_1-07-20-2016-DDSM-17695-1-cropped_images-02767-000000.dcm'
MASK = r'./Mass-Training_P_00059_LEFT_CC_1-07-21-2016-DDSM-38707-1-ROI_mask_images-82600-000000.dcm'

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