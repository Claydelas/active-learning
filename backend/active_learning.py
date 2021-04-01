import logging
import hashlib

from typing import Callable, Collection, Dict, Tuple, Union

from sklearn.base import BaseEstimator

from sio_server import Server
from learning import Learning, Vectorizer

import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np

from sklearn.model_selection import train_test_split

import modAL.uncertainty
from modAL.utils.data import drop_rows
from modAL.models import ActiveLearner

class ActiveLearning(Learning):
    def __init__(self,
                 estimator: BaseEstimator,
                 query_strategy: Callable[...,np.ndarray] = modAL.uncertainty.uncertainty_sampling,
                 dataset: DataFrame = None,
                 columns: Collection[Tuple[str, str]] = [('tweet', 'tweet')],
                 vectorizer: Vectorizer = None,
                 extra_processing: Callable[[DataFrame], DataFrame] = None,
                 start: bool = False,
                 target_score: float = 80.00,
                 n_queries: int = None):
        super().__init__(estimator, dataset, columns, vectorizer, extra_processing)
        assert callable(query_strategy), 'query_strategy must be callable'
        self.query_strategy = query_strategy
        self.target_score = target_score
        self.n_queries = n_queries
        # start webserver 
        if start: self.start()

    # utility function that provides a shortcut for building the active learning workflow
    def start(self,):
        self.process()
        self.learn_text_model()
        self.partition()
        self.split(self.labeled_pool)
        # initialise active learner model
        self.learner = ActiveLearner(
            estimator = self.estimator,
            query_strategy = self.query_strategy,
            X_training = self.X_train, y_training = self.y_train
        )
        self.auto_teach(self.n_queries)
        self.start_server()

    def split(self, pool: DataFrame, y: str = 'target'):
        # split into training and testing subsets
        # ensures at least 5 samples per class for initial training and testing
        labels = len(self.labeled_pool.index)
        if labels >= 100:
            X_unlabled, X_train_test = train_test_split(pool, random_state=42, test_size=0.2, stratify=pool[y])
            self.X_train_raw, self.X_test_raw, self.y_train, self.y_test = train_test_split(X_train_test.drop(y, axis=1), X_train_test[y], random_state=42, test_size=0.5, stratify=X_train_test[y])
            self.X_train, self.X_test = self.build_features(self.X_train_raw, self.columns), self.build_features(self.X_test_raw, self.columns)

            self.y_train = self.y_train.to_numpy()

            # unlabel 80%
            X_unlabled[y] = np.nan

            # merge with previously unlabled instances if any
            X_pool = pd.concat([X_unlabled, self.unlabeled_pool])

            self.X_pool_raw = X_pool.drop(y, axis=1)
            self.X_pool = self.build_features(self.X_pool_raw, self.columns)
            self.labeled_size = len(self.X_train_raw.index)
            self.dataset_size = len(self.dataset.index) - len(self.X_test_raw.index)
        elif labels >= 20:
            self.X_train_raw, self.X_test_raw, self.y_train, self.y_test = train_test_split(pool.drop(y, axis=1), pool[y], random_state=42, test_size=0.2, stratify=pool[y])
            self.X_train, self.X_test = self.build_features(self.X_train_raw, self.columns), self.build_features(self.X_test_raw, self.columns)
            self.X_pool_raw = self.unlabeled_pool.drop(y, axis=1)
            self.X_pool = self.build_features(self.X_pool_raw, self.columns)
            self.labeled_size = len(self.X_train_raw.index)
            self.dataset_size = len(self.dataset.index) - len(self.X_test_raw.index)
        else:
            self.X_pool_raw = self.unlabeled_pool.drop(y, axis=1)
            self.X_pool = self.build_features(self.X_pool_raw, self.columns)
            self.labeled_size = 0
            self.dataset_size = len(self.dataset.index)
            #raise Exception("Not enough labeled samples to fit classifier and generate test set")
        
    def auto_teach(self, queries = None):
        for _ in range(0, queries) if queries else range(0, self.X_pool.shape[0]):
            # retrieve most uncertain instance
            idx, sample = self.learner.query(self.X_pool)
            idx = int(idx)
            raw_idx = self.X_pool_raw.iloc[idx].name
            self.learner.teach(sample, np.array([self.labeled_pool.loc[raw_idx].target], dtype=int))
            self.labeled_size += 1
            # remove learned sample from pool
            self.X_pool_raw = self.X_pool_raw.drop(raw_idx)
            self.X_pool = drop_rows(self.X_pool, idx)
            # store accuracy metric after training
            self.accuracy_scores.append(dict(self.classification_report(self.X_test, self.y_test), labels=self.labeled_size))

    # utility function used to teach an active learner a new sample tweet from a json object {idx: i, hash: h, label: l}
    # after being learned, the tweet is removed from the sampling pool and the new model performance is recorded
    def teach(self, tweet: Dict[str, Union[int, str]], hashed = False):
        # extract data from tweet obj
        idx = int(tweet['idx'])
        label = int(tweet['label'])
        if self.X_pool_raw.empty: return
        text: str = self.X_pool_raw.iloc[idx].tweet
        y_new: np.ndarray = np.array([label], dtype=int)

        # fail early if hashes don't match (web app out of sync)
        if hashed and tweet['hash'] != hashlib.md5(text.encode()).hexdigest(): return

        # teach new sample
        #logging.info(f'-# Teaching instance: \n idx {idx}, \n label {label}, \n tweet: {text}, \n words: {self.X_pool_raw.iloc[idx].tweet_clean} #-')
        self.learner.teach(self.__get_row__(self.X_pool, idx), y_new)
        self.labeled_size += 1

        # remove learned sample from pool
        self.X_pool_raw = self.X_pool_raw.drop(self.X_pool_raw.iloc[idx].name)
        self.X_pool = drop_rows(self.X_pool, idx)

        # store accuracy metric after training
        self.accuracy_scores.append(dict(self.classification_report(self.X_test, self.y_test), labels=self.labeled_size))

    # starts an instance of the backend server used by the labeling web-app
    def start_server(self):
        server = Server(self, logging.getLogger())
        server.run()