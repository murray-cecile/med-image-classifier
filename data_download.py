'''
Initial data download menus

Step 1: Add desired images to the cart and download (I bulk selected the first 100)
Step 2: use NBIA data retriever to convert images from .tcia to .dcm (longest step)
Step 3: Flatten downloaded folders using the following linux command:
$ find /Users/Tammy/Documents/_MSCAPP/Winter_2020/Computer_Vision_MP/Test_Images/First_100/ -type f -exec sh -c 'for f do x=${f#./}; y="${x// /_}"; eval "mv ${x// /\ } ${y////-}"; done' {} +
Step 4: remove /Users/Tammy/Documents/_MSCAPP/Winter_2020/Computer_Vision_MP/Test_Images/First_100/ portion of directory name from all files using the following Linux command:
$ rename -- 's/-Users-Tammy-Documents-_MSCAPP-Winter_2020-Computer_Vision_MP-Test_Images-First_100--//' *
'''

import numpy as np
import png, os, pydicom
import matplotlib.pyplot as plt
import rasterio

PATH = r'/Users/Tammy/Documents/_MSCAPP/Winter_2020/Computer_Vision_MP/Test_Images/First_100'


# Run this in directory of interest
# Goal: create a json containing each file as well as characteristics
def open_img(single_file_path):
    '''
    '''
    ds = pydicom.dcmread(file)

    return ds


def plt_img(pydicom_img):
    '''
    '''
    plt.imhow(pydicom_img)


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