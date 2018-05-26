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
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np


def get_svm_best_parametres():
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
    svm = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                max_iter=-1, probability=False, random_state=None, shrinking=True,
                tol=0.001, verbose=False)
    # Training the model
    svm = svm.fit(X_train, y_train)

    print ('Studying parametres...')
    # Define range in parameters
    C = [1.0, 10.0, 100.0, 1000.0]
    gamma = [0.001, 0.0001]
    kernel = ['linear', 'rbf']


    # Searching the best parameters value
    print('Estudiando parametros')
    param_grid = {
        'C': C,
        'gamma': gamma,
        'kernel': ['rbf']
    }

    CV_svm = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5)
    CV_svm.fit(X, y)

    # PLot grid
    scores = [x[1] for x in CV_svm.grid_scores_]
    scores = np.array(scores).reshape(len(gamma), len(C))

    for ind, i in enumerate(gamma):
        plt.plot(C, scores[ind], label='gamma: ' + str(i))
    plt.legend()
    plt.xlabel('C')
    plt.ylabel('Mean score')
    plt.show()


    print(CV_svm.best_params_)


if __name__ == '__main__':
    get_svm_best_parametres()