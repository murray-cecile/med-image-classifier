#==============================================================================#
# EXPLORE CORRELATIONS BETWEEN FEATURES AND PATHOLOGY 
#
#==============================================================================#

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestRegressor, AdaBoostClassifier,
                              BaggingClassifier, GradientBoostingClassifier)



FEATURES = ['area', 'convex_area', 'eccentricity', 'equivalent_diameter', \
            'major_axis_length', 'minor_axis_length', 'orientation', \
            'perimeter', 'spiculationA', 'spiculationB', 'spiculationC', \
            'spiculationD']


def make_feat_pair_plots():

    for f1 in FEATURES:
        for f2 in FEATURES:

            if f1 == f2:
                continue
            else:
                plt.scatter(train[f1], train[f2], c=train['pathology'])
                plt.show()


def train_one_model(train, model):
    '''
    Trains the specified classifier
    Takes: train data and classifier object (with params set)
    Returns: trained classifier object
    '''

    outcome_label = 'pathology'
    x_train_no_id = train.drop('id', axis=1)
    x_train_no_id_outcome = x_train_no_id.drop(outcome_label, axis=1)
    x_train_label = train[outcome_label]
    features = x_train_no_id_outcome.columns

    return model.fit(x_train_no_id_outcome, x_train_label), features


if __name__ == "__main__":
    
    train = pd.read_csv("current_train.csv")

    # # correlation plot
    # feature_corr = train[FEATURES].corr()
    # f = plt.figure(figsize=(8, 8))
    # plt.matshow(feature_corr, fignum = f.number)
    # plt.xticks(range(len(FEATURES)), FEATURES, rotation=45)
    # plt.yticks(range(len(FEATURES)), FEATURES, rotation=45)
    # plt.show()

    # get feature importances from simple models
    model, features = train_one_model(train, Ridge(alpha=1))
    pd.DataFrame({"features": features, "coefs": model.coef_}).sort_values(by='coefs')

    model, features = train_one_model(train, DecisionTreeClassifier(depth=10))
    