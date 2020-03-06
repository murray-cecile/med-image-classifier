'''
Prediction Challenge Loop
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import pipeline as pipe

PATH = r'/Users/Tammy/Documents/_MSCAPP/Winter_2020/Computer_Vision_MP/med-image-classifier/raw_train/'
TRAIN_CSV = r'/Users/Tammy/Documents/_MSCAPP/Winter_2020/Computer_Vision_MP/med-image-classifier/metadata/training_labels.csv'


@ignore_warnings(category=ConvergenceWarning)
@ignore_warnings(category=FutureWarning)
def find_best_model(models, parameters_grid, x_train, outcome_label):
    '''
    Cross-validation to find the best model, given parameters
    '''
    results_df =  pd.DataFrame(columns=('model_name',
                                        'parameters',
                                        'auc',
                                        'time_to_run'))
    max_auc = 0
    best_model = ""
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

            # Calculate MSE using 5-fold cross validation
            # Change signs because scoring is negative MSE
            x_train_no_id = x_train.drop('id', axis=1)
            # print(x_train_no_id.head())
            # print(x_train_no_id.drop(outcome_label, axis=1).head())
            # print(x_train[outcome_label].head())
            scores = cross_val_score(estimator=model,
                                     X=x_train_no_id.drop(outcome_label, axis=1),
                                     y=x_train[outcome_label], # series or dataframe preferred?
                                     cv=5,
                                     scoring='roc_auc') #'neg_mean_squared_error'

            auc = scores.mean()
            time = timeit.default_timer() - start_time
            results_df.loc[(len(results_df))] = [model_key, parameter,
                                                 auc, time]

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
    

    # Fit best model and best parameter on full training dataset
    best_model.set_params(**best_parameter)
    best_model.fit(x_train_no_id.drop(outcome_label, axis=1),
                   x_train[outcome_label])

    return best_model


#### THIS IS NOT READY ####
def main():
    '''
    EXECUTE FULL LOOP.
    '''

    models = {'Tree': DecisionTreeRegressor(max_depth=10),
              'Lasso': Lasso(alpha=0.1),
              'Ridge': Ridge(alpha=.5),
              'Forest': RandomForestRegressor(max_depth=2)
              }

    parameters_grid = {'Tree': {'max_depth': [10, 20]},
                       'Lasso': {'alpha': [0.01, 0.1]},
                       'Ridge': {'alpha': [0.01, 0.1]},
                       'Forest': {'max_depth': [10, 20, 50]}
                       }

    outcome = 'pathology'
    train, test, test_ids = pipe.go() # CM implemented pipe.go, but hasn't tested

    best_model = find_best_model(models, parameters_grid, train, outcome)

    #Run predictions on test data and save file
    y_hats = best_model.predict(test)

    results  = pd.DataFrame(list(zip(test_ids, y_hats)),
                            columns =['id', 'y_hat'])

    results.to_csv('results.csv', index=False)

if __name__ == '__main__':
    # main()

    parser = argparse.ArgumentParser()
    parser.add_argument("-path", "--path", default = PATH, help = "Raw image file path")
    parser.add_argument("-train_csv", "--train_csv", default = TRAIN_CSV, help = "Training csv file path")
    args = parser.parse_args()


    models = {'Tree': DecisionTreeRegressor(max_depth=10),
              'Lasso': Lasso(alpha=0.1),
              'Ridge': Ridge(alpha=.5),
              'Forest': RandomForestRegressor(max_depth=2)
              }

    parameters_grid = {'Tree': {'max_depth': [10, 20]},
                       'Lasso': {'alpha': [0.01, 0.1]},
                       'Ridge': {'alpha': [0.01, 0.1]},
                       'Forest': {'max_depth': [10, 20, 50]}
                       }

    outcome = 'pathology'

    full_data = pipe.properties(path = args.path, train_csv = args.train_csv)

    train_data = full_data.drop(columns = ['patient_id', 'breast_density', \
    'left or right breast', 'image view', \
    'abnormality id', 'abnormality type', 'mass shape', 'mass margins', \
    'assessment', 'subtlety', 'image file path', 'cropped image file path', \
    'ROI mask file path'])
    train_data['pathology'] = train_data['pathology'].astype(float)

    train_data.to_csv("train_133.csv")
    # train_data = pd.read_csv("small_train.csv")    

    best_model = find_best_model(models, parameters_grid, train_data, outcome)

    