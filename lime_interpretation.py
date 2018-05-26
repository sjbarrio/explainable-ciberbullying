# Experiment 1. Script of sentiments analytics.
__author__      = 'Sergio Jiménez Barrio'
__copyright__   = "Copyright 2018, University of Seville"
__credits__     = ["Teodoro Álamo"]
__email__       = "sergio.jimbar@gmail.com"
__status__      = "BETA"
__version__     = "0.1"

from data_cleaning import CustomizedData
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import pickle

def interpretateWithLIME ():
    print ('Starting interpretation based on L.I.M.E. ...')
    dir = './machines/'
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
    Q = ["Today I see you with your boyfriend... doesnt matter. you're still a beautiful princess"]

    print ('Loading RF...')
    m = 'random_forest.sav'
    rfc = pickle.load(open(dir + m, 'rb'))

    print('Loading SVM...')
    m = 'svm.sav'
    svm = pickle.load(open(dir + m, 'rb'))

    print('Loading GBM...')
    m = 'gbt.sav'
    gbm = pickle.load(open(dir + m, 'rb'))

    print ('Getting interpretation with LIME...')
    c = make_pipeline(data.getVectorize(), svm)

    class_names = ['no_bullying', 'bullying']
    explainer = LimeTextExplainer(class_names=class_names)
    for index, post in enumerate(posts):
        print(c.predict_proba([post]))
        exp = explainer.explain_instance(post, c.predict_proba, num_features=6)
        exp.as_list()
        fig = exp.as_pyplot_figure()
        print ('Exporting ' + str(index) + '...')
        plt.savefig('./lime/lime_gbm_' + str(index))
        # Only for Q
        #plt.show(fig)



if __name__ == '__main__':
    interpretateWithLIME()


