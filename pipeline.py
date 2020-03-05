'''
Full pipeline
'''
import os
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt

from skimage.measure import label, regionprops, regionprops_table
from skimage.color import label2rgb
import preprocess as p


#FILE = r'./Mass-Training_P_00059_LEFT_CC_1-07-20-2016-DDSM-17695-1-cropped_images-02767-000000.dcm'
#MASK = r'./Mass-Training_P_00059_LEFT_CC_1-07-21-2016-DDSM-38707-1-ROI_mask_images-82600-000000.dcm'
PATH = r'/Users/Tammy/Documents/_MSCAPP/Winter_2020/Computer_Vision_MP/med-image-classifier/raw_train/'
TRAIN_CSV = r'/Users/Tammy/Documents/_MSCAPP/Winter_2020/Computer_Vision_MP/med-image-classifier/metadata/training_labels.csv'


def properties(path=PATH, train_csv=TRAIN_CSV):
    '''
    Calculate baseline set of features given a directory.
    '''
    list_of_files = os.listdir(path)
    df = pd.DataFrame()

    labels = pd.read_csv(train_csv)
    labels['id'] = labels['id'].str.split('/').str[0]

    for file in list_of_files:
        try:
            full_path = path + file
            filled_img, original = p.go(full_path)
            label_image = label(filled_img)
            props = regionprops_table(label_image,
                                      properties=['area', # Num of pixes in region
                                                  'convex_area', # Num pixels in smallest convex polygon that encloses region
                                                  'eccentricity', # Eccentricity of the ellipse that has the same second-moments as the region
                                                  'equivalent_diameter', # Diameter of circle with same area as region
                                                  'major_axis_length', # Length of major axis
                                                  'minor_axis_length', # Length of minor axis
                                                  #'moments_central', # Central moments up to 3rd order
                                                  'orientation',
                                                  'perimeter',])
            for key in props: 
                props[key] = float(props[key]) 
            props['id'] = original.PatientID
            df = df.append(props, ignore_index=True)
        
        except:
            print('Could not process: ', file)

    full_data = df.join(labels.set_index('id'), on='id')
    # Change benign without callback to benign
    full_data.loc[full_data['pathology'] == 'BENIGN_WITHOUT_CALLBACK', 'pathology'] = 'BENIGN'
    full_data.loc[full_data['pathology'] == 'BENIGN', 'pathology'] = 0
    full_data.loc[full_data['pathology'] == 'MALIGNANT', 'pathology'] = 1

    return full_data



















