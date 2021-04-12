#%% ----------------------------------------------------------------imports
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../backend'))

from sio_server import Server

import gensim.downloader as api
from gensim.models.doc2vec import Doc2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd

from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
#%% ----------------------------------------------------------------datasets
t_davidson = pd.read_pickle("data/processed/t-davidson_processed.pkl")
#%% ----------------------------------------------------------------word embeddings
glove = api.load('glove-twitter-25')
#%% ----------------------------------------------------------------doc2vec
doc2vec = Doc2Vec(vector_size=50, min_count=2, epochs=10)
#%% ----------------------------------------------------------------tfidf
tfidf = TfidfVectorizer(
    ngram_range=(1, 3),
    use_idf=True,
    smooth_idf=False,
    norm=None,
    decode_error='replace',
    max_features=10000,
    min_df=5,
    max_df=0.75
    )
#%% ----------------------------------------------------------------
options = {
    'classifiers': [
        {'name': 'Logistic Regression', 'classifier': LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)},
        {'name': 'Linear SVM', 'classifier': LinearSVC(loss='hinge', class_weight='balanced', random_state=42)},
        {'name': 'Random Forest', 'classifier': RandomForestClassifier(random_state=42, class_weight='balanced')},
        {'name': 'Ada Boost', 'classifier': AdaBoostClassifier(base_estimator=LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42), random_state=42)}],
    'datasets': [
        {'name': 'T Davidson Hate Speech/Offesnive Language',
         'df': t_davidson,
         'targets': [
             {'val': 0,'name': 'hate speech'},
             {'val': 1,'name': 'offensive language'},
             {'val': 2,'name': 'neither'}]
        }],
    'vectorizers': [
        {'name': 'TF IDF Vectorizer', 'vectorizer': tfidf},
        {'name': 'GloVe Word Embeddings', 'vectorizer': glove},
        {'name': 'Doc2Vec Paragraph Embeddings', 'vectorizer': doc2vec}],
    'features': [
        {'name': 'text', 'cols': [('tweet', 'tweet')]},
        {'name': 'user', 'cols': []},
        {'name': 'stats', 'cols': [('emoji_count', 'numeric'),
                                   ('polarity', 'numeric'),
                                   ('subjectivity', 'numeric'),
                                   ('hashtag_count', 'numeric'),
                                   ('mentions_count', 'numeric'),
                                   ('words_count', 'numeric'),
                                   ('char_count', 'numeric'),
                                   ('url_count', 'numeric'),
                                   ('is_retweet', 'bool'),]}],
    'query_strategies': [
        {'name': 'Uncertainty Sampling', 'strategy': uncertainty_sampling},
        {'name': 'Entropy Sampling', 'strategy': entropy_sampling},
        {'name': 'Margin Sampling', 'strategy': margin_sampling}],
    }
server = Server(options=options).run()