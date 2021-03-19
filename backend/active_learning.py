import logging, sys
import modAL.uncertainty
import random
from modAL.utils.data import data_hstack, drop_rows, retrieve_rows
from modAL.models import ActiveLearner
from pandas.core.frame import DataFrame
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import preprocess as pre
import numpy as np
import gensim
import hashlib
from sklearn.model_selection import train_test_split
from sio_server import Server

class ActiveLearning():
    def __init__(self,
                 estimator,
                 query_strategy = modAL.uncertainty.uncertainty_sampling,
                 dataset: DataFrame = None,
                 columns = [('tweet', 'tweet')],
                 vectorizer = None,
                 start = False
    ):
        assert callable(query_strategy), 'query_strategy must be callable'

        logging.basicConfig(handlers=[logging.FileHandler('server.log', 'a', 'utf-8')], level=logging.DEBUG, format='[%(asctime)s] %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

        self.query_strategy = query_strategy
        self.estimator = estimator

        # default dataset
        if dataset is None: dataset = pd.read_csv("../data/dataset.csv", sep='\t', index_col=0)
        # append target column to dataframe if it doesn't exist
        if not 'target' in dataset.columns: dataset['target'] = np.nan
        # rename the attribute containing tweet text to "tweet"
        for col, type in columns:
            if type == 'tweet':
                dataset.rename(columns={col: 'tweet'}, inplace=True)
        self.dataset = dataset

        # rename the column containing tweet text to "tweet" (for feature extraction)
        self.columns = [('tweet', type) if type == 'tweet' else (col,type) for col, type in columns]
        self.accuracy_scores = []

        # default vectorizer
        if vectorizer is None:
            self.vectorizer = TfidfVectorizer(ngram_range=(1,3))
        else: self.vectorizer = vectorizer

        # start webserver 
        if start: self.start()

    # utility function that provides a shortcut for building the active learning workflow
    def start(self,):
        self.process()
        self.learn_text_model()
        self.partition()
        self.setup_pool()
        # initialise active learner model
        self.learner = ActiveLearner(
            estimator = self.estimator,
            query_strategy = self.query_strategy,
            #X_training = self.X_train, y_training = self.y_train
        )
        self.start_server()

    # extracts features from text and prepares it for vectorization
    def process(self,):
        clean_columns = []
        for col, type in self.columns:
            if (type == 'tweet'):
                # extract textual features
                pre.feature_extract(self.dataset, col)

                # clean train set
                pre.process(self.dataset, col)

                # remove duplicate tweets
                self.dataset = self.dataset.drop_duplicates(subset=['tweet_clean'], ignore_index=True)

                # rename the preprocessed column from name -> name_clean
                clean_columns.append((f'{col}_clean',type))
            elif (type == 'text'):
                # clean train set
                pre.process(self.dataset, col)

                # rename the preprocessed column from name -> name_clean
                clean_columns.append((f'{col}_clean',type))
            else:
                # keep the column name
                clean_columns.append((col,type))
        self.columns = clean_columns
        return self.dataset

    # trains a vectorizer on a set of documents
    def learn_text_model(self, vectorizer = None, documents = 'tweet_clean'):
        if vectorizer is None:
            vectorizer = self.vectorizer 
        if isinstance(vectorizer, TfidfVectorizer):
            # learn TF-IDF language model
            vectorizer.fit(self.dataset[documents])
        elif isinstance(self.vectorizer, gensim.models.doc2vec.Doc2Vec):
            # learn Doc2Vec language model
            train_corpus = [gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(row[documents]), [index]) for index, row in self.dataset.iterrows()]
            vectorizer.build_vocab(train_corpus)
            vectorizer.train(train_corpus, total_examples=vectorizer.corpus_count, epochs=vectorizer.epochs)
        else: raise Exception("undefined behaviour for specified vectorizer")
        return vectorizer

    # partitions dataset into unlabeled, training and testing subsets
    def partition(self, y='target'):
        # partition dataset into labeled and unlabled samples
        self.labeled_pool = self.dataset[self.dataset.target.notnull()].reset_index(drop=True)
        self.unlabeled_pool = self.dataset[self.dataset.target.isnull()].reset_index(drop=True)

        self.dataset_size = len(self.dataset.index)
        self.labeled_size = len(self.labeled_pool.index)

        # split into training and testing subsets
        # ensures at least 5 samples per class for initial training and testing
        if self.labeled_size >= 100:
            X_unlabled, X_train_test = train_test_split(self.labeled_pool, random_state=42, test_size=0.2, stratify=self.labeled_pool[y])
            X_unlabled[y] = np.nan
            self.unlabeled_pool = pd.concat([self.unlabeled_pool, X_unlabled], ignore_index=True)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_train_test.drop(y, axis=1), X_train_test[y], random_state=42, test_size=0.5, stratify=X_train_test[y])
            return self.X_train, self.X_test, self.y_train, self.y_test
        elif self.labeled_size >= 20:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.labeled_pool.drop(y, axis=1), self.labeled_pool[y], random_state=42, test_size=0.5, stratify=self.labeled_pool[y])
            return self.X_train, self.X_test, self.y_train, self.y_test
        else:
            pass
            #raise Exception("Not enough labeled samples to fit classifier and generate test set")

    # prepares features for fitting with a classifier
    def setup_pool(self):
        self.X_raw, self.y = self.unlabeled_pool.drop('target', axis=1), self.unlabeled_pool.target
        self.X = self.build_features(self.X_raw, self.columns)
        return self.X_raw, self.X, self.y

    # builds and stacks feature matrices to obtain a single matrix used for sampling and training
    def build_features(self, pool, columns):
        blocks = []
        for column, type in columns:
            if type == 'text' or type == 'tweet':
                blocks.append(self.__vectorize__(pool[column]))
            if type == 'numeric':
                # TODO: scale numeric features
                blocks.append(pool[column].values.reshape(-1,1))
            if type == 'bool':
                blocks.append(pool[column].apply(lambda val: 1 if val == True else 0).values.reshape(-1,1))
        return data_hstack(blocks)

    # utility function used to teach an active learner a new sample tweet from a json object {idx: i, hash: h, label: l}
    # after being learned, the tweet is removed from the sampling pool and the new model performance is recorded
    def teach(self, tweet):
        # extract data from tweet obj
        idx = int(tweet['idx'])
        label = int(tweet['label'])
        text = self.X_raw.iloc[idx].tweet
        y_new = np.array([label], dtype=int)

        # fail early if hashes don't match (web app out of sync)
        if tweet['hash'] != hashlib.md5(text.encode()).hexdigest(): return

        # teach new sample
        logging.info(f'-# Teaching instance: \n idx {idx}, \n label {label}, \n tweet: {text}, \n words: {self.X_raw.iloc[idx].tweet_clean} #-')
        self.learner.teach(self.__get_row__(self.X, idx), y_new)
        self.labeled_size += 1

        # remove learned sample from pool
        self.X_raw = self.X_raw.drop(idx).reset_index(drop=True)
        self.y = self.y.drop(idx).reset_index(drop=True)
        self.X = drop_rows(self.X, idx)

        # store accuracy metric after training
        # TODO: obtain test sets
        # accuracy_scores.append({'queries': self.labeled_size, 'score': learner.score(self.X_test, self.y_test)*100})
        self.accuracy_scores.append({'queries': self.labeled_size, 'score': random.random()*100})

    # starts an instance of the backend server used by the labeling web-app
    def start_server(self):
        server = Server(self, logging.getLogger())
        server.run()

    # utility function that provides a uniform method for vectorizing text via different vectorizers
    def __vectorize__(self, documents, vectorizer = None):
        if vectorizer is None:
            vectorizer = self.vectorizer
        if isinstance(vectorizer, TfidfVectorizer):
            return vectorizer.transform(documents)
        elif isinstance(vectorizer, gensim.models.doc2vec.Doc2Vec):
            return np.array([vectorizer.infer_vector(gensim.utils.simple_preprocess(x)) for x in documents])

    # utility function that returns a row as a 2d np array regardless of matrix format
    def __get_row__(self, feature_matrix, idx): 
        if isinstance(feature_matrix, np.ndarray):
            return retrieve_rows(feature_matrix, [idx])
        else:
            return retrieve_rows(feature_matrix, idx)