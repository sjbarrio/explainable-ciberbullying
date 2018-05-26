# Experiment 1. Script of sentiments analytics.
__author__      = 'Sergio Jiménez Barrio'
__copyright__   = "Copyright 2018, University of Seville"
__credits__     = ["Teodoro Álamo"]
__email__       = "sergio.jimbar@gmail.com"
__status__      = "BETA"
__version__     = "0.1"

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from string import punctuation

################
# Configuration
################
# Configuration parameters to spanish stem
# nltk.download("all")
english_stopwords = stopwords.words('english')
stemmer = SnowballStemmer('english')

# Extends no words with interrogative a exclamations symbols
non_words = list(punctuation)
non_words.extend(map(str, range(10)))

# Choose spanish language to tokenizer
stemmer = SnowballStemmer('english')

# #############
# Functions
###############
def stem_tokens(tokens, stemmer):
    """
    Convert each word in the root  word
    :param tokens: Tweet Tokenized
    :param stemmer: Stemmer to use
    :return: Tokens with root words
    """
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    """
    Tokenizer a text with stem applied
    :param text: text to tokenizer
    :return: stems
    """

    # Tokenizer a text
    text = ''.join([c for c in text if c not in non_words])
    tokens = word_tokenize(text)

    # stem
    try:
        stems = stem_tokens(tokens, stemmer)
    except Exception as e:
        print(e)
        print(text)
        stems = ['']
    return stems


class CustomizedData():
    def __init__(self):
        ##########################
        #     EXTRACT DATA       #
        ##########################
        # Get corpus
        self.df_corpus = pd.read_csv('./data/formspring_compensated_data.csv', encoding='utf-8', sep=';')
        ####################
        # COUNT VECTORIZER
        ####################
        print ('Creating vectorize...')
        # Create vectorize with CountVectorizer
        self.vectorizer = CountVectorizer(
                analyzer='word',
                tokenizer=tokenize,
                lowercase=True,
                stop_words=english_stopwords,
                min_df=1,
                max_df=500,
                ngram_range=(1, 1),
                max_features=1000
        )

        # Vectorizer dataset
        corpus_data_features = self.vectorizer.fit_transform(self.df_corpus.text)
        corpus_data_features_nd = corpus_data_features.toarray()

        # Vectorizer binary dataset
        self.corpus_data_features_bin = self.vectorizer.fit_transform(self.df_corpus.text)
        self.corpus_data_features_nd_bin = self.corpus_data_features_bin.toarray()

        print ('Class initialized!')

    def getVectorize (self):
        return self.vectorizer

    def getX (self):
        return self.corpus_data_features_nd_bin

    def getY (self):
        return self.df_corpus['ciberbullying']

    def get_columns (self):
        return self.vectorizer.get_feature_names()

    def prepare_data(self, text):
        df_test = pd.DataFrame(columns=['text'], data=text)
        df = self.df_corpus.append(df_test)
        corpus_data_features_bin = self.vectorizer.fit_transform(df.text)
        return (corpus_data_features_bin.toarray()[len(self.df_corpus):len(df)])
