import logging, sys
from typing import Callable, Collection, Dict, Tuple, Union

from sklearn.base import BaseEstimator

import preprocess as pre

import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import gensim
import gensim.downloader as api

from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from modAL.utils.data import data_hstack, retrieve_rows

Vectorizer = Union[TfidfVectorizer, gensim.models.doc2vec.Doc2Vec, gensim.models.word2vec.Word2Vec, gensim.models.keyedvectors.KeyedVectors]

# base class for learning processes
class Learning():
    def __init__(self,
                 estimator: BaseEstimator,
                 dataset: DataFrame = None,
                 columns: Collection[Tuple[str, str]] = [('tweet', 'tweet')],
                 vectorizer: Vectorizer = None,
                 preprocess: bool = False,
                 extra_processing: Callable[[DataFrame], DataFrame] = None,
                 start: bool = False):

        logging.basicConfig(handlers=[logging.FileHandler('server.log', 'a', 'utf-8')], level=logging.DEBUG, format='[%(asctime)s] %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

        self.estimator = estimator

        # default dataset
        if dataset is None: dataset = pd.read_csv("../data/dataset.csv", sep='\t', index_col=0)
        # append target column to dataframe if it doesn't exist
        if not 'target' in dataset.columns: dataset['target'] = np.nan
        # rename the attribute containing tweet text to "tweet"
        for col, type in columns:
            if type == 'tweet':
                dataset.rename(columns={col: 'tweet'}, inplace=True)

        # execute any aditional processing defined as a callback function (extra dataset-specific processing)
        if extra_processing is not None and callable(extra_processing): 
            dataset = extra_processing(dataset)
        self.dataset = dataset

        # rename the column containing tweet text to "tweet" (for feature extraction)
        self.columns = [('tweet', type) if type == 'tweet' else (col,type) for col, type in columns]
        self.accuracy_scores: Collection[Dict[str, Union[int, float]]] = []

        # default vectorizer
        if vectorizer is None:
            self.vectorizer = TfidfVectorizer(ngram_range=(1,3))
        else: self.vectorizer = vectorizer

        self.preprocess = preprocess

        # start
        if start:
            self.start()

    def start(self):
        if self.preprocess: self.process()
        self.learn_text_model()
        self.split()
        self.fit(X=self.X_train, y=self.y_train)


    # extracts features from text and prepares it for vectorization
    def process(self,) -> DataFrame:
        clean_columns: Collection[Tuple[str, str]] = []
        for col, type in self.columns:
            if (type == 'tweet'):
                # extract textual features
                pre.feature_extract(self.dataset, col)
                # clean train set
                pre.process(self.dataset, col)
                # remove duplicate tweets
                self.dataset = self.dataset.drop_duplicates(subset=['tweet_clean'])
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
    def learn_text_model(self, vectorizer: Vectorizer = None, documents: str = 'tweet_clean') -> Vectorizer:
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
        elif isinstance(self.vectorizer, gensim.models.word2vec.Word2Vec):
            # load pre-trained Word2Vec model
            vectorizer = api.load('glove-twitter-50')
            self.vectorizer = vectorizer
        else: raise Exception("undefined behaviour for specified vectorizer")
        return vectorizer


    def split(self, pool: DataFrame = None, y: str = 'target', test_size = 0.1):

        if pool is None: pool = self.dataset[self.dataset.target.notnull()]

        labels = len(pool.index)
        if labels >= 100:
            train, test = train_test_split(pool, random_state=42, test_size=test_size)

            self.X_train_raw = train.drop(y, axis=1)
            self.X_train = self.build_features(self.X_train_raw, self.columns)
            self.y_train = train[y]

            self.X_test_raw = test.drop(y, axis=1)
            self.X_test = self.build_features(self.X_test_raw, self.columns)
            self.y_test = test[y]
        elif labels >= 20:
            self.X_train_raw, self.X_test_raw, self.y_train, self.y_test = train_test_split(pool.drop(y, axis=1), pool[y], random_state=42, test_size=test_size)
            self.X_train, self.X_test = self.build_features(self.X_train_raw, self.columns), self.build_features(self.X_test_raw, self.columns)
        else:
            raise Exception("Not enough labeled samples to fit classifier and generate test set")
        self.labeled_size = len(self.X_train_raw.index)
        self.dataset_size = len(self.dataset.index) - len(self.X_test_raw.index)

    # builds and stacks feature matrices to obtain a single matrix used for sampling and training
    def build_features(self, pool: DataFrame, columns: Collection[Tuple[str, str]]):
        blocks = []
        for column, type in columns:
            if type == 'text' or type == 'tweet':
                blocks.append(self.__vectorize__(pool[column]))
            elif type == 'numeric':
                # TODO: scale numeric features
                blocks.append(pool[column].values.reshape(-1,1))
            elif type == 'bool':
                blocks.append(pool[column].apply(lambda val: 1 if val == True else 0).values.reshape(-1,1))
        return data_hstack(blocks)

    # utility function that provides a uniform method for vectorizing text via different vectorizers
    def __vectorize__(self, documents, vectorizer: Vectorizer = None):
        if vectorizer is None:
            vectorizer = self.vectorizer
        if isinstance(vectorizer, TfidfVectorizer):
            return vectorizer.transform(documents)
        elif isinstance(vectorizer, gensim.models.doc2vec.Doc2Vec):
            return np.array([vectorizer.infer_vector(gensim.utils.simple_preprocess(x)) for x in documents])
        elif isinstance(self.vectorizer, gensim.models.keyedvectors.KeyedVectors):
            wset = set(self.vectorizer.wv.index2word)
            return np.array([self._avg_w2v_(x, wset) for x in documents])

    def _avg_w2v_(self, doc, wset):
        featureVec = np.zeros((self.vectorizer.vector_size,), dtype="float32")
        nwords = 0
        words = doc.split()
        for word in words:
            if word in wset:
                nwords = nwords+1
                featureVec = np.add(featureVec, self.vectorizer[word])

        if nwords>0:
            featureVec = np.divide(featureVec, nwords)
        return featureVec

    # utility function that returns a row as a 2d np array regardless of matrix format
    def __get_row__(self, feature_matrix, idx: int): 
        if isinstance(feature_matrix, np.ndarray):
            return retrieve_rows(feature_matrix, [idx])
        else:
            return retrieve_rows(feature_matrix, idx)

    def fit(self, X, y):
        self.estimator.fit(X=X, y=y)
        return self.estimator

    def f1_score(self, X, y, average = 'micro'):
        return f1_score(y_pred=self.estimator.predict(X), y_true=y, average=average)

    def classification_report(self, X, y):
        return classification_report(y_pred=self.estimator.predict(X), y_true=y, output_dict=True)