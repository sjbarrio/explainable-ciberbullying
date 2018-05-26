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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from scipy import sparse
from sklearn import tree
import pickle
import sklearn
from sklearn.decomposition import sparse_encode


def create_bests_machine():
    dir = './machines/'
    print ('Creating machines with best parametres...')
    print ('Getting data...')
    data = CustomizedData()
    """
    X = data.getX()
    y = data.getY()
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33, random_state=42)
    """
    X_train = data.getX()
    y_train = data.getY()


    ###################################
    ##        RANDOM FOREST          ##
    ###################################
    print('Creating Random Forest...')
    m = 'random_forest.sav'
    rfc = RandomForestClassifier(n_jobs=-1, max_features='sqrt', n_estimators=100, oob_score=True, max_depth=50)
    rfc = rfc.fit(X_train, y_train)
    pickle.dump(rfc, open(dir + m, 'wb'))

    """
    print ('Random Forest Matrix Confusion:')
    predictions = rfc.predict(X_test)
    conf_matrix = sklearn.metrics.confusion_matrix(y_test, predictions)
    print (conf_matrix)
    print ('Random Forest Score:')
    score = sklearn.metrics.accuracy_score(y_test, predictions)
    print(score)
    """

    ###################################
    ##              SVM              ##
    ###################################
    print('Creating SVM...')
    m = 'svm.sav'
    svm = SVC(C=100.0, gamma=0.001, probability=True, kernel='rbf')
    svm = svm.fit(sparse.coo_matrix(X_train), y_train)
    pickle.dump(svm, open(dir + m, 'wb'))
    """
    print('SVM Matrix Confusion:')
    predictions = svm.predict(X_test)
    conf_matrix = sklearn.metrics.confusion_matrix(y_test, predictions)
    print(conf_matrix)
    print('SVM:')
    score = sklearn.metrics.accuracy_score(y_test, predictions)
    print(score)
    """

    ###################################
    ##              GBM              ##
    ###################################
    print('Creating Gradient Boosting...')
    m = 'gbt.sav'
    gbm= GradientBoostingClassifier(max_features='log2', n_estimators=125, max_depth=10)
    gbm = rfc.fit(X_train, y_train)
    pickle.dump(gbm, open(dir + m, 'wb'))
    """
    print ('Gradient Boosting Matrix Confusion:')
    predictions = gbm.predict(X_test)
    conf_matrix = sklearn.metrics.confusion_matrix(y_test, predictions)
    print (conf_matrix)
    print ('Gradient Boosting Score:')
    score = sklearn.metrics.accuracy_score(y_test, predictions)
    print(score)
    """


if __name__ == '__main__':
    create_bests_machine()