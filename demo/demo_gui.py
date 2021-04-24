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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#%% ----------------------------------------------------------------datasets
t_davidson = pd.read_pickle("data/processed/t-davidson_processed.pkl")
personal = pd.read_pickle("data/processed/personal_processed.pkl")
#%% ----------------------------------------------------------------word embeddings
glove = api.load('glove-twitter-25')
#%% ----------------------------------------------------------------doc2vec
doc2vec = Doc2Vec(vector_size=50, min_count=2, epochs=10)
#%% ----------------------------------------------------------------tfidf
tfidf = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=8000,
    min_df=1,
    max_df=0.75
    )
#%% ----------------------------------------------------------------
options = {
    'classifiers': [
        {'name': 'Logistic Regression', 'classifier': LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)},
        {'name': 'Linear SVM', 'classifier': SVC(class_weight='balanced', kernel='linear', probability=True, random_state=42)},
        {'name': 'Random Forest', 'classifier': RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs= -1)},
        {'name': 'K Nearest Neighbors', 'classifier': KNeighborsClassifier(n_neighbors=2, n_jobs= -1)}],
    'datasets': [
        {'name': 'T Davidson Hate Speech/Offensive Language',
         'df': t_davidson,
         'targets': [
             {'val': 0,'name': 'hate speech'},
             {'val': 1,'name': 'offensive language'},
             {'val': 2,'name': 'neither'}]
        },
        {'name': 'Project Dataset',
         'df': personal,
         'targets': [
             {'val': 0,'name': 'non-malicious'},
             {'val': 1,'name': 'malicious'}]
        }],
    'vectorizers': [
        {'name': 'TF IDF Vectorizer', 'vectorizer': tfidf},
        {'name': 'GloVe Word Embeddings', 'vectorizer': glove},
        {'name': 'Doc2Vec Paragraph Embeddings', 'vectorizer': doc2vec}],
    'features': [
        {'name': 'text', 'cols': [('tweet', 'tweet')]},
        {'name': 'user', 'cols': [('user_is_verified', 'bool'),
                                  ('user_posts', 'numeric'),
                                  ('user_likes','numeric'),
                                  ('user_followers', 'numeric'),
                                  ('user_friends', 'numeric')]},
        {'name': 'stats', 'cols': [('emoji_count', 'numeric'),
                                   ('polarity', 'numeric'),
                                   ('subjectivity', 'numeric'),
                                   ('hashtag_count', 'numeric'),
                                   ('mentions_count', 'numeric'),
                                   ('words_count', 'numeric'),
                                   ('char_count', 'numeric'),
                                   ('url_count', 'numeric'),
                                   ('is_retweet', 'bool'),
                                   ('tweet_likes', 'numeric'),
                                   ('tweet_retweets', 'numeric'),
                                   ('tweet_is_quote', 'bool')]}],
    'query_strategies': [
        {'name': 'Uncertainty Sampling', 'strategy': uncertainty_sampling},
        {'name': 'Entropy Sampling', 'strategy': entropy_sampling},
        {'name': 'Margin Sampling', 'strategy': margin_sampling}],
    }
server = Server(options=options).run()