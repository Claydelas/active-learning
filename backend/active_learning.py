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
                 learn_vectorizer: bool = False,
                 preprocess: bool = False,
                 extra_processing: Callable[[DataFrame], DataFrame] = None,
                 start: bool = False,
                 target_score: float = 80.00,
                 n_queries: int = None):
        super().__init__(estimator, dataset, columns, vectorizer, learn_vectorizer, preprocess, extra_processing, start)
        assert callable(query_strategy), 'query_strategy must be callable'
        self.query_strategy = query_strategy
        self.target_score = target_score
        self.n_queries = n_queries

        self.learner = ActiveLearner(
            estimator = self.estimator,
            query_strategy = self.query_strategy
        )

        # start webserver 
        if start:
            self.start_server()


    # utility function that provides a shortcut for building the active learning workflow
    def start(self, auto:bool = False):
        super().start()
        if auto: self.auto_teach(self.n_queries)
        self.start_server()


    def split(self, pool: DataFrame = None, y: str = 'target', test_size = 0.1, train_size = 0.01):

        if pool is None: pool = self.dataset[self.dataset.target.notnull()]
        unlabeled_frame = self.dataset[self.dataset.target.isnull()]

        labels = len(pool.index)
        if labels >= 100:
            pool_train, test = train_test_split(pool, random_state=42, test_size=test_size)
            self.X_test = self.build_features(test, self.columns)
            self.y_test = test[y]

            unlabeled_pool, train = train_test_split(pool_train, random_state=42, test_size=train_size, stratify=pool_train[y])

            self.X_train = self.build_features(train, self.columns)
            self.y_train = train[y].to_numpy()

            # merge with previously unlabled instances if any
            X_pool = pd.concat([unlabeled_pool, unlabeled_frame])

            self.idx_map = dict(enumerate(X_pool.index))
            self.X_pool = self.build_features(X_pool, self.columns)

            self.labeled_size = len(train.index)
            self.dataset_size = len(self.dataset.index) - len(test.index)
        elif labels >= 20:
            train, test, self.y_train, self.y_test = train_test_split(pool, pool[y], random_state=42, test_size=0.5, stratify=pool[y])
            self.X_train, self.X_test = self.build_features(train, self.columns), self.build_features(test, self.columns)
            
            self.idx_map = dict(enumerate(unlabeled_frame.index))
            self.X_pool = self.build_features(unlabeled_frame, self.columns)

            self.labeled_size = len(train.index)
            self.dataset_size = len(self.dataset.index) - len(test.index)
        else:
            self.idx_map = dict(enumerate(unlabeled_frame.index))
            self.X_pool = self.build_features(unlabeled_frame, self.columns)

            self.labeled_size = labels
            self.dataset_size = len(self.dataset.index)
            #raise Exception("Not enough labeled samples to fit classifier and generate test set")
        
    def auto_teach(self, queries = None):
        for _ in range(0, queries) if queries else range(0, self.X_pool.shape[0]):
            # retrieve most uncertain instance
            idx, sample = self.learner.query(self.X_pool)
            idx = int(idx)
            dataset_idx = self.idx_map.get(idx)
            self.learner.teach(sample, np.array([self.dataset.loc[dataset_idx].target], dtype=int))
            self.labeled_size += 1
            # remove learned sample from pool
            self.X_pool = drop_rows(self.X_pool, idx)
            # store accuracy metric after training
            self.accuracy_scores.append(dict(self.classification_report(self.X_test, self.y_test), labels=self.labeled_size))


    # utility function used to teach an active learner a new sample tweet from a json object {idx: i, hash: h, label: l}
    # after being learned, the tweet is removed from the sampling pool and the new model performance is recorded
    def teach(self, tweet: Dict[str, Union[int, str]], hashed = False):
        # extract data from tweet obj
        idx = int(tweet['idx'])
        label = int(tweet['label'])
        if self.X_pool.shape[0] == 0: return
        dataset_idx = self.idx_map.get(idx)
        text: str = self.dataset.loc[dataset_idx].tweet
        y_new: np.ndarray = np.array([label], dtype=int)

        # fail early if hashes don't match (web app out of sync)
        if hashed and tweet['hash'] != hashlib.md5(text.encode()).hexdigest(): return

        # teach new sample
        #logging.info(f'-# Teaching instance: \n idx {idx}, \n label {label}, \n tweet: {text}, \n words: {self.dataset.loc[dataset_idx].tweet_clean} #-')
        self.learner.teach(self._get_row_(self.X_pool, idx), y_new)
        self.dataset.at[dataset_idx, 'target'] = label
        self.labeled_size += 1

        # remove learned sample from pool
        self.X_pool = drop_rows(self.X_pool, idx)

        # store accuracy metric after training
        self.accuracy_scores.append(dict(self.classification_report(self.X_test, self.y_test), labels=self.labeled_size))


    # starts an instance of the backend server used by the labeling web-app
    def start_server(self):
        server = Server(self, logging.getLogger())
        server.run()


    def fit(self, X, y):
        self.learner.fit(X=X, y=y)
        return self.learner