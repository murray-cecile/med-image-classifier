'''
Final Script: Prediction Challenge Loop

Runs the full classification loop. Inputs include training and testing set
directory paths, as well as training and testing labels saved in a CSV.
The script will print a summary of all evaluated models as well as the
highest AUC acheived.
'''
import pandas as pd
import sklearn
import datetime
import timeit
import argparse

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ParameterGrid
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import pipeline as pipe


@ignore_warnings(category=ConvergenceWarning)
@ignore_warnings(category=FutureWarning)
def find_best_model(models, parameters_grid, x_train, outcome_label):
    '''
    Cross-validation to find the best model, given parameters.

    Input:
        models (dict): dictionary initializing each scikit-learn model
        parameters_grid (dict): dictionary setting hyperparameter values
        x_train (df): processed training feature dataframe
        outcome_label (str): name of outcome label

    Output:
        best_model (obj): best model fit on full training data
    '''
    results_df =  pd.DataFrame(columns=('model_name',
                                        'parameters',
                                        'auc',
                                        'time_to_run'))
    max_auc = 0
    best_model = ""
    best_parameter = ""
    start_time = timeit.default_timer()

    for model_key in models:
        print("Starting " + model_key + " at " + str(datetime.datetime.now()))
        model = models[model_key]
        parameter_values = parameters_grid[model_key]
        
        for parameter in ParameterGrid(parameter_values):
            s = timeit.default_timer()
            model.set_params(**parameter)

            # Calculate AUC using 5-fold cross validation
            x_train_no_id = x_train.drop('id', axis=1)
            scores = cross_val_score(estimator=model,
                                     X=x_train_no_id.drop(outcome_label, axis=1),
                                     y=x_train[outcome_label],
                                     cv=5,
                                     scoring='roc_auc')
            auc = scores.mean()
            time = timeit.default_timer() - start_time
            results_df.loc[len(results_df)] = [model_key, parameter, auc, time]

            # Update "winner"
            if (auc > max_auc):
                max_auc = auc
                best_model = model
                best_parameter = parameter
                best_model_type = model_key

    elapsed = timeit.default_timer() - start_time

    print(results_df)
    print("Highest AUC " + str(max_auc))
    print("Best Model " + str(best_model))
    print("Best Parameter " + str(best_parameter))
    print('Total Time: ', elapsed)
    
    # Fit best model & best parameters on full training dataset
    best_model.set_params(**best_parameter)
    best_model.fit(x_train_no_id.drop(outcome_label, axis=1),
                   x_train[outcome_label])

    return best_model


def main(args):
    '''
    Full loop execution.

    Input:
        args (dict): dictionary of command line arguments

    Return:
        train (df): pre-processed training dataframe [CHANGE LATER]
    '''
    models = {'Tree': DecisionTreeRegressor(max_depth=10),
              'Lasso': Lasso(alpha=0.1),
              'Ridge': Ridge(alpha=.5),
              'Forest': RandomForestRegressor(max_depth=2),
              'SVM': SVC(C=1, kernel='rbf')
              }

    parameters_grid = {'Tree': {'max_depth': [10, 20]},
                       'Lasso': {'alpha': [0.01, 0.1]},
                       'Ridge': {'alpha': [0.01, 0.1]},
                       'Forest': {'max_depth': [10, 20, 50]},
                       'SVM': {'C': [0.1, 1, 10]}
                       }

    outcome = 'pathology'

    train, test, test_ids = pipe.go(args.train, args.train_csv, args.test, args.test_csv)

    train.to_csv("current_train.csv")

    best_model = find_best_model(models, parameters_grid, train, outcome)

    #Run predictions on test data and save file
    if args.test:
        y_hats = best_model.predict(test.drop(columns = ['id']))

        results  = pd.DataFrame(list(zip(test_ids, y_hats)),
                                columns =['id', 'y_hat'])

        results.to_csv('results.csv', index=False)
    
    return train

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-train", "--train", default = "", help = "Training image file path")
    parser.add_argument("-train_csv", "--train_csv", default = "", help = "Training csv file path")
    parser.add_argument("-test", "--test", default= "", help = "Test image file path")
    parser.add_argument("-test_csv", "--test_csv", default="", help = "Testing csv file path")

    args = parser.parse_args()


    train_data = main(args)