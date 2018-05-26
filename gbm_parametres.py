# Experiment 1. Script of sentiments analytics.
__author__      = 'Sergio Jiménez Barrio'
__copyright__   = "Copyright 2018, University of Seville"
__credits__     = ["Teodoro Álamo"]
__email__       = "sergio.jimbar@gmail.com"
__status__      = "BETA"
__version__     = "0.1"

# Imports
from data_cleaning import CustomizedData
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

def get_gbm_best_parametres():
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
    gbc = GradientBoostingClassifier(max_features='sqrt', n_estimators=500, max_depth=100)
    # Training the model
    gbc = gbc.fit(X_train, y_train)

    print ('Studying parametres...')
    # Define range in parameters
    n_estimators = [1, 2, 5, 10, 25, 50, 75, 100, 125, 200]
    max_depth = [2, 5, 10, 25, 50, 75, 100]

    # Searching the best parameters value
    param_grid = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'max_features': ['log2']
    }

    CV_gbc = GridSearchCV(estimator=gbc, param_grid=param_grid, cv=5)
    CV_gbc.fit(X, y)

    # PLot grid
    scores = [x[1] for x in CV_gbc.grid_scores_]
    scores = np.array(scores).reshape(len(max_depth), len(n_estimators))

    for ind, i in enumerate(max_depth):
        plt.plot(n_estimators, scores[ind], label='Max Depth: ' + str(i))
    plt.legend()
    plt.xlabel('n_estimators')
    plt.ylabel('Mean score')
    plt.show()

    print(CV_gbc.best_params_)


if __name__ == '__main__':
    get_gbm_best_parametres()