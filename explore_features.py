#==============================================================================#
# EXPLORE CORRELATIONS BETWEEN FEATURES AND PATHOLOGY 
#
#==============================================================================#

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

FEATURES = ['area', 'convex_area', 'eccentricity', 'equivalent_diameter', \
            'major_axis_length', 'minor_axis_length', 'orientation', \
            'perimeter', 'spiculationA', 'spiculationB', 'spiculationC', \
            'spiculationD']

if __name__ == "__main__":
    
    train = pd.read_csv("current_train.csv")

    # # correlation plot
    feature_corr = train[FEATURES].corr()
    f = plt.figure(figsize=(8, 8))
    plt.matshow(feature_corr, fignum = f.number)
    plt.xticks(range(len(FEATURES)), FEATURES, rotation=45)
    plt.yticks(range(len(FEATURES)), FEATURES, rotation=45)
    plt.show()

    # for f1 in FEATURES:
    #     for f2 in FEATURES:

    #         if f1 == f2:
    #             continue
    #         else:
    #             plt.scatter(train[f1], train[f2], c=train['pathology'])
    #             plt.show()