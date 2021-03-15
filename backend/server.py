import os
import glob
import re
from time import time
import demoji
from modAL.utils.data import data_hstack, drop_rows, retrieve_rows
import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

import random
import hashlib

import getopt, sys

from modAL.models import ActiveLearner
import modAL.uncertainty
import modAL.batch

from sklearn.model_selection import train_test_split

import logging
import threading
from datetime import datetime
from time import sleep

from flask import Flask, request
from flask_socketio import SocketIO

logging.basicConfig(handlers=[logging.FileHandler('server.log', 'a', 'utf-8')], level=logging.DEBUG, format='[%(asctime)s] %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

try:
    args, vals = getopt.getopt(sys.argv[1:], 'd:s:av:', ['dataset=', 'sampling=', 'auto', 'vectorizer='])
except getopt.error as err:
    logging.error(str(err))
    sys.exit(2)

# defaults
DATASET = "../data/dataset.csv"
QUERY_STRATEGY = modAL.uncertainty.uncertainty_sampling
MANUAL = True
VECTORIZER = 'tfidf'

# config arg parsing
for arg, val in args:
    if arg in ("-d", "--dataset"):
        if os.path.isfile(val):
            DATASET = val
            logging.info(f'Working with dataset @{val}')
    elif arg in ("-s", "--sampling"):
        options = {
            "uncertainty": modAL.uncertainty.uncertainty_sampling,
            "margin": modAL.uncertainty.margin_sampling,
            "entropy": modAL.uncertainty.entropy_sampling,
            "batch": modAL.batch.uncertainty_batch_sampling
        }
        QUERY_STRATEGY = options.get(val, modAL.uncertainty.uncertainty_sampling)
        logging.info(f'Querying with @{str(QUERY_STRATEGY)}')
    elif arg in ("-a", "--auto"):
        logging.info('Semi-supervised label querying is enabled.')
        MANUAL = False
    elif arg in ("-v", "--vectorizer"):
        if val in ("tfidf", "w2v", "d2v"):
            logging.info(f'Using {val} Vectorizer.')
            VECTORIZER = val

# load dataset into a data frame
df = pd.read_csv(DATASET, sep='\t', index_col=0)

# bootstrap nlp libraries
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('punkt')
nltk.download('wordnet')
wl = WordNetLemmatizer()
demoji.download_codes()

# contractions regex
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", 
                   "can't've": "cannot have", "'cause": "because", "could've": "could have", 
                   "couldn't": "could not", "couldn't've": "could not have","didn't": "did not", 
                   "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
                   "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", 
                   "he'd": "he would", "he'd've": "he would have", "he'll": "he will", 
                   "he'll've": "he will have", "he's": "he is", "how'd": "how did", 
                   "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 
                   "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
                   "I'll've": "I will have","I'm": "I am", "I've": "I have", 
                   "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 
                   "i'll've": "i will have","i'm": "i am", "i've": "i have", 
                   "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
                   "it'll": "it will", "it'll've": "it will have","it's": "it is", 
                   "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
                   "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
                   "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
                   "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
                   "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                   "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
                   "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
                   "she's": "she is", "should've": "should have", "shouldn't": "should not", 
                   "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
                   "this's": "this is",
                   "that'd": "that would", "that'd've": "that would have","that's": "that is", 
                   "there'd": "there would", "there'd've": "there would have","there's": "there is", 
                       "here's": "here is",
                   "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
                   "they'll've": "they will have", "they're": "they are", "they've": "they have", 
                   "to've": "to have", "wasn't": "was not", "we'd": "we would", 
                   "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
                   "we're": "we are", "we've": "we have", "weren't": "were not", 
                   "what'll": "what will", "what'll've": "what will have", "what're": "what are", 
                   "what's": "what is", "what've": "what have", "when's": "when is", 
                   "when've": "when have", "where'd": "where did", "where's": "where is", 
                   "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
                   "who's": "who is", "who've": "who have", "why's": "why is", 
                   "why've": "why have", "will've": "will have", "won't": "will not", 
                   "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
                   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                   "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                   "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
                   "you'll've": "you will have", "you're": "you are", "you've": "you have" }
contracts = re.compile(r'\b(' + '|'.join(contraction_mapping.keys()) + r')\b')

# extract textual features in-place (for external comparative datasets)
def feature_extract():
    df['emoji_count'] = df.tweet.apply(lambda tweet: len(demoji.findall_list(tweet)))
    df['polarity'] = df.tweet.apply(lambda tweet: TextBlob(tweet).polarity)
    df['subjectivity'] = df.tweet.apply(lambda tweet: TextBlob(tweet).subjectivity)
    df['hashtag_count'] = df.tweet.str.count('#')
    df['mentions_count'] = df.tweet.str.count('@')
    df['words_count'] = df.tweet.str.split().str.len()
    df['char_count'] = df.tweet.str.len()
    df['url_count'] = df.tweet.str.count('https?://\S+')
    df['is_retweet'] = df.tweet.apply(lambda tweet: 1 if re.search('[Rr][Tt].@\S+', tweet) else 0)

# tweet normalisation/cleaning
def clean(tweet):
    # drop all urls
    no_urls = re.sub(r'(((https?:\s?)?(\\\s*/\s*)*\s*t\.co\s*(s*\\\s*/\s*)*\S+)|https?://\S+)', '', tweet)
    # transform emojis to their description
    new = demoji.replace_with_desc(string=no_urls, sep='\"')
    # fix html encoded chars
    new = BeautifulSoup(new, 'lxml').get_text()
    # remove retweet tags
    no_retweets = re.sub("[Rr][Tt].@\S+", " ", new)
    # lower case all words
    lower_case = no_retweets.lower()
    # fix negated words with apostrophe
    negations_fix = contracts.sub(lambda x: contraction_mapping[x.group()], lower_case)
    # remove all mentions
    no_mentions = re.sub("@\S+", " ", negations_fix)
    # remove all special characters
    letters_only = re.sub("[^a-zA-Z]", " ", no_mentions)
    # tokenize
    words = [x for x in nltk.word_tokenize(letters_only) if len(x) > 1]
    # remove stop words
    no_stopwords = list(filter(lambda l: l not in stop_words, words))
    # lemmatize words
    lemmas = [wl.lemmatize(t) for t in no_stopwords]
    return (" ".join(lemmas)).strip()

# extract textual features
feature_extract()

# clean train set
df['clean'] = df.tweet.apply(clean)
df = df.drop_duplicates(subset=['clean'], ignore_index=True)
if not 'target' in df.columns: df['target'] = np.nan

if VECTORIZER == 'tfidf': 
    # learn TF-IDF language model
    tfidf = TfidfVectorizer(ngram_range=(1,3))
    tfidf.fit(df['clean'])

import gensim
if VECTORIZER == 'd2v':
    # learn Doc2Vec language model
    train_corpus = [gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(row['clean']), [index]) for index, row in df.iterrows()]
    doc_model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
    doc_model.build_vocab(train_corpus)
    doc_model.train(train_corpus, total_examples=doc_model.corpus_count, epochs=doc_model.epochs)

# learn Word2Vec language model
if VECTORIZER == 'w2v':
    #TODO: w2v averaged
    pass

def vectorize(documents):
    if VECTORIZER == 'tfidf':
        return tfidf.transform(documents)
    elif VECTORIZER == 'd2v':
        return np.array([doc_model.infer_vector(gensim.utils.simple_preprocess(x)) for x in documents])

def get_row(feature_matrix, idx): 
    if isinstance(feature_matrix, np.ndarray):
        return retrieve_rows(feature_matrix, [idx])
    else:
        return retrieve_rows(feature_matrix, idx)

def build_features(pool, columns):
    blocks = []
    for column, type in columns:
        if type == 'text':
            blocks.append(vectorize(pool[column]))
        if type == 'numeric':
            # TODO: scale numeric features
            blocks.append(pool[column].values.reshape(-1,1))
        if type == 'bool':
            blocks.append(pool[column].apply(lambda val: 1 if val == True else 0).values.reshape(-1,1))
    return data_hstack(blocks)

# partition dataset if possible
labeled_pool = df[df.target.notnull()]
labeled_pool.reset_index(drop=True, inplace=True)
unlabeled_pool = df[df.target.isnull()]
unlabeled_pool.reset_index(drop=True, inplace=True)

# note dataset size
dataset_size = len(df.index)
labeled_size = len(labeled_pool.index)

# split into training and testing subsets if possible 
## X_train, X_test, y_train, y_test = train_test_split(labeled_pool.drop('target', axis=1), labeled_pool.target, random_state=42)

# set up learning pool
X_pool_raw, y_pool = unlabeled_pool.drop('target', axis=1), unlabeled_pool.target
X_pool_features = vectorize(X_pool_raw['clean'])
# TODO: pull features from args
# replaces X_pool_features in final implementation
X_all_features = build_features(X_pool_raw, [
    ('clean', 'text'), # vectorize on clean
    ('user_followers', 'numeric'), # scale user_followers
    ('user_verified', 'bool') # transform bools to 0/1s
    ])

# initialise active learner model
# TODO: allow supplying of estimator + h-params as command args
classifier = KNeighborsClassifier(n_neighbors=1)
learner = ActiveLearner(
    estimator = classifier,
    query_strategy = QUERY_STRATEGY
)

# accuracy_scores = [learner.score(X_test, y_test)]
accuracy_scores = []

def teach(tweet):

    global X_pool_raw, X_pool_features, y_pool, labeled_size, accuracy_scores

    # extract data from tweet obj
    idx = int(tweet['idx'])
    label = int(tweet['label'])
    text = X_pool_raw.iloc[idx].tweet
    y_new = np.array([label], dtype=int)

    # fail early if hashes don't match (web app out of sync)
    if tweet['hash'] != hashlib.md5(text.encode()).hexdigest(): return

    # teach new sample
    logging.info(f'-# Teaching instance: \n idx {idx}, \n label {label}, \n tweet: {text}, \n words: {X_pool_raw.iloc[idx].clean} #-')
    learner.teach(get_row(X_pool_features, idx), y_new)
    labeled_size += 1

    # remove learned sample from pool
    X_pool_raw = X_pool_raw.drop(idx).reset_index(drop=True)
    y_pool = y_pool.drop(idx).reset_index(drop=True)
    X_pool_features = drop_rows(X_pool_features, idx)
    
    # store accuracy metric after training
    # TODO: obtain test sets
    # accuracy_scores.append({'queries': labeled_size, 'score': learner.score(X_test, y_test)*100})
    accuracy_scores.append({'queries': labeled_size, 'score': random.random()*100})

def init():
    # retrieve most uncertain instance
    query_idx, query_sample = learner.query(X_pool_features)
    idx = int(query_idx)

    sio.emit('init', {
            'idx': idx,
            'text': X_pool_raw.iloc[idx].tweet,
            'uncertainty': modAL.uncertainty.classifier_uncertainty(classifier=classifier, X=query_sample)[0],
            'series': accuracy_scores,
            'strategy': 'uncertainty',
            'labeled_size': labeled_size,
            'dataset_size': dataset_size,
            'score': accuracy_scores[-1]['score'] if accuracy_scores else 0,
            'target': 80.00
            })

def query():
    # retrieve most uncertain instance
    query_idx, query_sample = learner.query(X_pool_features)
    idx = int(query_idx)

    sio.emit('query', {
            'idx': idx,
            'text': X_pool_raw.iloc[idx].tweet,
            'uncertainty': modAL.uncertainty.classifier_uncertainty(classifier=classifier, X=query_sample)[0],
            'labeled_size': labeled_size,
            'series': accuracy_scores[-1],
            'score': accuracy_scores[-1]['score']
            })

if MANUAL:
    logging.info('Starting AL server')
    app = Flask('Active Learning')
    sio = SocketIO(app, cors_allowed_origins='*')

    @app.route('/')
    def index():
        return 'Server running on localhost:5000.'

    @sio.on('connect')
    def connect():
        logging.info(f'Client connected: {request.sid}')
        init()

    @sio.on('disconnect')
    def disconnect():
        logging.info(f'Client disconnected: {request.sid}')

    @sio.on('refresh')
    def refresh():
        logging.info(f'{request.sid} requested refresh.')
        init()

    @sio.on('label')
    def label(tweet):
        teach(tweet)
        query()

    # Start Web Server + Socket.IO
    # thread = threading.Thread(target=lambda: sio.run(app)).start()
    sio.run(app)
