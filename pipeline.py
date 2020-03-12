#============================================================================#
# BUILD TRAINING AND TESTING FEATURE SET
#============================================================================#

'''
Inputs include training and testing set directory paths, as well as training
and testing labels saved to a CSV. The script will return featurized testing
and training data.
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
from sklearn.utils.testing import ignore_warnings

import preprocess as p
import create_features as cf

pd.set_option('display.max_columns', 500)
@ignore_warnings(category=FutureWarning)
def properties(img_dir, csv_path):
    '''
    Calculates a pre-processed feature set given a directory of images.

    Input:
        img_dir (str): image directory
        csv_path (str): location of training labels stored in a CSV

    Output:
        full_data (df): a pandas dataframe containing patient_id, features,
                        and pathology
    '''
    list_of_files = os.listdir(img_dir)
    df = pd.DataFrame()

    labels = pd.read_csv(csv_path)
    labels['id'] = labels['cropped image file path'].str.split('/').str[0] 

    # drop extraneous metadata columns
    labels.drop(columns = ['patient_id', 'breast_density', \
    'left or right breast', 'image view', \
    'abnormality id', 'abnormality type', 'mass shape', 'mass margins', \
    'assessment', 'subtlety', 'image file path', 'cropped image file path', \
    'ROI mask file path'], inplace=True)

    for file in list_of_files:
        try:
            full_path = img_dir + file
            filled_img, original = p.go(full_path)
            label_image = label(filled_img)
            props = regionprops_table(label_image,
                                        properties=['area', # Num of pixels in region
                                                    'convex_area', # Num pixels in smallest convex polygon that encloses region
                                                    'eccentricity', # Eccentricity of the ellipse that has the same second-moments as the region
                                                    'equivalent_diameter', # Diameter of circle with same area as region
                                                    'major_axis_length', # Length of major axis
                                                    'minor_axis_length', # Length of minor axis
                                                    'perimeter'])

            # manual feature generation from create_features
            manual_features = cf.make_all_features(original, filled_img)
            for f in manual_features.keys():
                props[f] = manual_features[f]

            for key in props: 
                props[key] = float(props[key]) 
            # include id
            props['id'] = original.PatientID
            df = df.append(props, ignore_index=True)
    
        except Exception as e:
            print('Could not process: ', file)
            print(e)

    
    # Optional: standardize numeric columns
    # features = ['area', 'convex_area', 'eccentricity', 'equivalent_diameter',\
    #            'major_axis_length', 'minor_axis_length', 'perimeter',\
    #            'spiculationA', 'spiculationB', 'spiculationC', 'spiculationD',\
    #            'spiculationRA', 'spiculationRB', 'spiculationRC', 'spiculationRD',\
    #            'circularity', 'iou', 'hough', 'snake', 'gabor']

    full_data = df.join(labels.set_index('id'), on='id')
    # full_data[features] = minmax_scale(full_data[features])
    

    # Change benign without callback to benign
    full_data.loc[full_data['pathology'] == 'BENIGN_WITHOUT_CALLBACK', 'pathology'] = 'BENIGN'
    full_data.loc[full_data['pathology'] == 'BENIGN', 'pathology'] = 0
    full_data.loc[full_data['pathology'] == 'MALIGNANT', 'pathology'] = 1
    full_data['pathology'] = full_data['pathology'].astype(float)

    return full_data


def go(train_path, train_csv, test_path=None, test_csv=None):
    '''
    Creates training and testing features.

    Input:
        train_path (str): training images path
        train_csv (str): training labels csv
        test_path (str): testing images path
        test_csv (str): testing labels csv

    Return:
        train_data (df): pandas dataframe, including id, features, and label
        test_data (df): pandas dataframe, including id, features, and label.
                        Note: if parameters set to None, will return None.
    '''
    test_data = None
    train_data = properties(train_path, train_csv)

    if test_path and test_csv:
        test_data = properties(test_path, test_csv)
    
    return train_data, test_data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-train", "--train", default="", help="Training image file path")
    parser.add_argument("-train_csv", "--train_csv", default="", help="Training csv file path")
    parser.add_argument("-test", "--test", default="", help = "Test image file path")
    parser.add_argument("-test_csv", "--test_csv", default="", help ="Testing csv file path")
    args = parser.parse_args()

    try:
        train, test, test_labels = go(args.train, args.train_csv)
    except Exception as e:
        print(e)
