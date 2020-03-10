'''
Full pipeline
'''
import os
import argparse
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt

from skimage.measure import label, regionprops, regionprops_table
from skimage.color import label2rgb
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import minmax_scale

import preprocess as p
import create_features as cf

pd.set_option('display.max_columns', 500)

#FILE = r'./Mass-Training_P_00059_LEFT_CC_1-07-20-2016-DDSM-17695-1-cropped_images-02767-000000.dcm'
#MASK = r'./Mass-Training_P_00059_LEFT_CC_1-07-21-2016-DDSM-38707-1-ROI_mask_images-82600-000000.dcm'
PATH = r'/Users/Tammy/Documents/_MSCAPP/Winter_2020/Computer_Vision_MP/med-image-classifier/raw_train/'
TRAIN_CSV = r'/Users/Tammy/Documents/_MSCAPP/Winter_2020/Computer_Vision_MP/med-image-classifier/metadata/training_labels.csv'


def properties(img_dir=PATH, csv_path=TRAIN_CSV):
    '''
    Calculate baseline set of features given the training directory.
    '''
    list_of_files = os.listdir(img_dir)
    df = pd.DataFrame()

    labels = pd.read_csv(csv_path)
    labels['id'] = labels['cropped image file path'].str.split('/').str[0] 

    for file in list_of_files:
        try:
            full_path = img_dir + file
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

            # manual feature generation from create_features
            manual_features = cf.make_all_features(original, filled_img) 
            # print(manual_features)
            for f in manual_features.keys():
                props[f] = manual_features[f]

            for key in props: 
                props[key] = float(props[key]) 
            props['id'] = original.PatientID

            df = df.append(props, ignore_index=True)
        
        except Exception as e:
            print('Could not process: ', file)
            print(e)

    # Standardize numeric columns
    df[['area', 'convex_area', 'eccentricity', 'equivalent_diameter',\
        'major_axis_length', 'minor_axis_length', 'orientation', 'perimeter']] =\
        minmax_scale(df[['area', 'convex_area', 'eccentricity', 'equivalent_diameter',\
        'major_axis_length', 'minor_axis_length', 'orientation', 'perimeter']])

    full_data = df.join(labels.set_index('id'), on='id')
    # Change benign without callback to benign
    full_data.loc[full_data['pathology'] == 'BENIGN_WITHOUT_CALLBACK', 'pathology'] = 'BENIGN'
    full_data.loc[full_data['pathology'] == 'BENIGN', 'pathology'] = 0
    full_data.loc[full_data['pathology'] == 'MALIGNANT', 'pathology'] = 1

    full_data['pathology'] = full_data['pathology'].astype(float)

    return full_data



def go(train_path, train_csv, test_path = None, test_csv = None):
    '''
    Creates train data, test data, and test labels for the prediction loop
    Takes: file paths to image directories and train/test metadata csvs
    Returns: train, test, and test labels
    '''

    train_data = properties(train_path, train_csv)
    
    # drop extraneous columns - maybe change this in properties() instead?
    train_data = train_data.drop(columns = ['patient_id', 'breast_density', \
    'left or right breast', 'image view', \
    'abnormality id', 'abnormality type', 'mass shape', 'mass margins', \
    'assessment', 'subtlety', 'image file path', 'cropped image file path', \
    'ROI mask file path'])

    if test_path and test_csv:
        test_data = properties(test_path, test_csv)
        test_data = test_data.drop(columns = ['patient_id', 'breast_density', \
        'left or right breast', 'image view', \
        'abnormality id', 'abnormality type', 'mass shape', 'mass margins', \
        'assessment', 'subtlety', 'image file path', 'cropped image file path', \
        'ROI mask file path'])

        # extract the labels from the test data
        test_labels = test_data['pathology']
        test_data = test_data.drop(columns = ['pathology'])
    else:
        test_data = None
        test_labels = None
    
    return train_data, test_data, test_labels


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-path", "--path", default = PATH, help = "Raw image file path")
    parser.add_argument("-train_csv", "--train_csv", default = TRAIN_CSV, help = "Training csv file path")
    args = parser.parse_args()

    try:
        train, test, test_labels = go(args.path, args.train_csv)
    except Exception as e:
        print(e)