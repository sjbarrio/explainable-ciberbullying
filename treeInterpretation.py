# Experiment 1. Script of sentiments analytics.
__author__      = 'Sergio Jiménez Barrio'
__copyright__   = "Copyright 2018, University of Seville"
__credits__     = ["Teodoro Álamo"]
__email__       = "sergio.jimbar@gmail.com"
__status__      = "BETA"
__version__     = "0.1"

from data_cleaning import CustomizedData
import numpy as np
from sklearn import tree
import pandas as pd
from sklearn.pipeline import make_pipeline
from treeinterpreter import treeinterpreter as ti
import matplotlib.pyplot as plt
import sklearn
import pickle


def tree_interpretation():
    print ('Starting interpretation based on contributions...')
    dir = './machines/'
    test = "You're a fucking bitch"
    data = CustomizedData()
    posts = ["You're ugly and a sack of shit",  # 1
             "Today I see you with your boyfriend... doesnt matter. you're still a fake bitch",  # 1
             "You suck",  # 1
             "Do you like the new movie?",  # 0
             "suck my big dick",  # 1
             "hello mr gay fuck",  # 1
             "I want fuck u",  # 1
             "When is your brithday?",  # 0
             "I now you are an asskisser, always sucking dicks",  # 1
             "I'm trying help you",  # 0
             "Do you have a favorite article of clothing?",  # 0
             "Where do you wish you were right now?",  # 0
             "Good look tomorrow in the exam!",  # 0
             "You're so cute",  # 0
             "Do you have any favorite football player?",  # 0
             "How would you describe yourself?",  # 0
             "you are gay",  # 1
             "Go to the hell, piece of shit",  # 1
             "Stop eating, fat!",  # 1
             "You are nasty"]  # 1


    print ('Loading RF...')
    m = 'random_forest.sav'
    rfc = pickle.load(open(dir + m, 'rb'))

    print('Loading GBM...')
    m = 'gbt.sav'
    gbm = pickle.load(open(dir + m, 'rb'))


    # Variable importance
    df_variales = pd.DataFrame(columns=['variable', 'importance'])
    df_variales['variable'] = data.get_columns()
    df_variales['importance'] = gbm.feature_importances_
    print(df_variales)

    """
    # export tree plot
    print('Exporting trees...')
    i_tree = 0
    for tree_in_forest in rfc.estimators_:
        with open('./rf_trees/tree_' + str(i_tree) + '.dot', 'w') as my_file:
            my_file = tree.export_graphviz(tree_in_forest, out_file=my_file, feature_names=data.get_columns())
        i_tree = i_tree + 1
    i_tree = 0
    for tree_in_forest in gbm.estimators_:
        with open('./gbm_trees/tree_' + str(i_tree) + '.dot', 'w') as my_file:
            my_file = tree.export_graphviz(tree_in_forest, out_file=my_file, feature_names=data.get_columns())
        i_tree = i_tree + 1
    """

    # Export to excel

    writer = pd.ExcelWriter('variable_importance_gbm.xlsx')
    df_variales.to_excel(writer, 'variable_importance')
    writer.save()

    col = data.get_columns()
    list_= []
    list_n = []
    list_p = []
    for i in col:
        list_n.append(i + '_neg')
        list_p.append(i + '_pos')
        list_.append(i + '_neg')
        list_.append(i + '_pos')
    # Interpretation with tree interpreter
    X = data.prepare_data(posts)
    prediction, biases, contributions = ti.predict(gbm, X)
    df_results = pd.DataFrame(columns=['prediction_neg', 'prediction_pos', 'bia_neg', 'bia_pos'] + list_n + list_p)
    # Set contributions table
    df_results.prediction_neg = prediction[:,0]
    df_results.prediction_pos = prediction[:,1]
    df_results.bia_neg = biases[:, 0]
    df_results.bia_pos = biases[:, 1]
    for idx, feature in enumerate(list_n):
        df_results[feature] = contributions[:,idx,0]
    for idx, feature in enumerate(list_p):
        df_results[feature] = contributions[:,idx,1]

    # Re-order dataframe
    cols = ['prediction_neg', 'prediction_pos', 'bia_neg', 'bia_pos'] + list_
    df_results = df_results[cols]
    # Get only bullying contributions
    contr_bullying = df_results[list_]
    # Trans df
    contr_bullying = contr_bullying.transpose()
    print(contr_bullying)
    # Export to excel
    writer = pd.ExcelWriter('tree_interpreter_gbm.xlsx')
    df_results.to_excel(writer, 'gbm_tree_intr_results')
    contr_bullying.to_excel(writer, 'gbm_bullying_contribution')
    writer.save()




if __name__ == '__main__':
    tree_interpretation()