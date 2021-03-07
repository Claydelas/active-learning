import os
import glob
import re
from time import time
import demoji
import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from textblob import TextBlob
#from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt


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

logging.basicConfig(filename='server.log', level=logging.DEBUG, format='[%(asctime)s] %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

try:
    args, vals = getopt.getopt(sys.argv, 'd:s:a', ['dataset=', 'sampling=', 'auto'])
except getopt.error as err:
    logging.error(str(err))
    sys.exit(2)

DATASET = "../data/dataset.csv"
QUERY_STRATEGY = modAL.uncertainty.uncertainty_sampling
MANUAL = True

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
        
if MANUAL:
    app = Flask('Active Learning')
    sio = SocketIO(app, cors_allowed_origins='*')


    @app.route('/')
    def index():
        return 'Server running on localhost:5000.'


    # Start Web Server + Socket.IO
    thread = threading.Thread(target=lambda: sio.run(app)).start()


    @sio.on('connect')
    def connect():
        logging.info(f'Client connected: {request.sid}')
        # TODO: on connect emit data samples for labeling


    @sio.on('disconnect')
    def disconnect():
        logging.info(f'Client disconnected: {request.sid}')


    @sio.on('refresh')
    def refresh():
        # TODO: Refresh AL process
        print('refresh')


    @sio.on('label')
    def label(tweet):
        # TODO: Teach new label
        print(f'idx {tweet.idx}, label {tweet.label}')


# normal script execution below:
logging.info('Starting AL process')

# load dataset into a data frame
df = pd.read_csv(DATASET, sep='\t', index_col=0)

# bootstrap nlp libraries
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('punkt')
nltk.download('wordnet')
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
    return (" ".join(no_stopwords)).strip()


# extract textual features
feature_extract()

# clean train set
df['clean'] = df.tweet.apply(clean)
df = df.drop_duplicates(subset=['clean'], ignore_index=True)
if not 'target' in df.columns: df['target'] = np.nan
logging.info(df.count())

# partition dataset if possible
labeled_pool = df[df.target.notnull()]
unlabeled_pool = df[df.target.isnull()]

# TODO: Utilise all features, not only clean text
# split into training and testing subsets if possible 
X_train, X_test, y_train, y_test = train_test_split(labeled_pool.clean, labeled_pool.target)
# set up learning pool
X_pool, y_pool = unlabeled_pool.clean, unlabeled_pool.target

# initialise active learner model
# TODO: allow supplying of estimator + h-params as command args
learner = ActiveLearner(
    estimator=LogisticRegression(),
    query_strategy=QUERY_STRATEGY,
    X_training=X_train, y_training=y_train
)

# learning loop
n_queries = 20
accuracy_scores = [learner.score(X_test, y_test)]
for i in range(n_queries):
    
    # retrieve most uncertain instance
    query_idx, query_inst = learner.query(X_pool)
    
    # display this query in the front end
    # TODO
    
    # retrieve new label
    # TODO
    y_new = np.array([int(input())], dtype=int) #input() is the new label after the user has answered the query
    
    # teach new sample
    learner.teach(query_inst.reshape(1, -1), y_new)
    
    # remove learned sample from pool
    X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx, axis=0)
    
    # store accuracy metric after training
    accuracy_scores.append(learner.score(X_test, y_test))
    
# sample real-time clock for testing
while True:
    sio.emit('tweet', {'idx': 0, 'text': datetime.now().strftime("%H:%M:%S")})
    sleep(1)
