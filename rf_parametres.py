# Experiment 1. Script of sentiments analytics.
__author__      = 'Sergio Jiménez Barrio'
__copyright__   = "Copyright 2017, University of Seville"
__credits__     = ["Teodoro Alamo"]
__email__       = "sergio.jimbar@gmail.com"
__status__      = "BETA"
__version__     = "0.1"

# Imports
from data_cleaning import CustomizedData
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np


def get_rf_best_parametres():
    data =  CustomizedData()
    print('Creating train and test dataset...')
    # Define X and Y
    X = data.getX()
    y = data.getY()
    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33, random_state=42)

    # Model
    print ('Generating model...')
    rfc = RandomForestClassifier(n_jobs=-1, max_features='sqrt', n_estimators=500, oob_score=True, max_depth=100)
    # Training the model
    rfc = rfc.fit(X_train, y_train)

    print ('Studying parametres..')
    # Define range in parameters
    n_estimators = [1, 2, 5, 10, 25, 50, 75, 100, 125, 200, 500]
    max_depth = [2, 5, 10, 25, 50, 75, 100]

    # Searching the best parameters value
    param_grid = {
        'n_estimators': n_estimators,

        'max_depth': max_depth,
        'max_features': ['log2']
    }

    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    CV_rfc.fit(X, y)

    # PLot grid
    scores = [x[1] for x in CV_rfc.grid_scores_]
    scores = np.array(scores).reshape(len(max_depth), len(n_estimators))

    for ind, i in enumerate(max_depth):
        plt.plot(n_estimators, scores[ind], label='Max Depth: ' + str(i))
    plt.legend()
    plt.xlabel('n_estimators')
    plt.ylabel('Mean score')
    plt.show()

    print(CV_rfc.best_params_)


if __name__ == '__main__':
    get_rf_best_parametres()