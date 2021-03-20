import logging
import random
import hashlib

from typing import Callable, Collection, Dict, Tuple, Union

from sklearn.base import BaseEstimator

from backend.sio_server import Server
from backend.learning import Learning, Vectorizer

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
                 target_score: float = 80.00):
        super().__init__(estimator, dataset, columns, vectorizer, extra_processing)
        assert callable(query_strategy), 'query_strategy must be callable'
        self.query_strategy = query_strategy
        self.target_score = target_score
        # start webserver 
        if start: self.start()

    # utility function that provides a shortcut for building the active learning workflow
    def start(self,):
        self.process()
        self.learn_text_model()
        self.partition()
        self.split(self.labeled_pool)
        self.setup_pool()
        # initialise active learner model
        self.learner = ActiveLearner(
            estimator = self.estimator,
            query_strategy = self.query_strategy,
            #X_training = self.X_train, y_training = self.y_train
        )
        self.start_server()

    def split(self, pool: DataFrame, y: str = 'target'):
        # split into training and testing subsets
        # ensures at least 5 samples per class for initial training and testing
        if self.labeled_size >= 100:
            X_unlabled, X_train_test = train_test_split(pool, random_state=42, test_size=0.2, stratify=pool[y])
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_train_test.drop(y, axis=1), X_train_test[y], random_state=42, test_size=0.5, stratify=X_train_test[y])

            # unlabel 80%
            X_unlabled[y] = np.nan

            # merge with previously unlabled instances if any
            self.X_raw = pd.concat([self.unlabeled_pool, X_unlabled], ignore_index=True)

            self.X_raw = self.X_raw.drop(y, axis=1)
            self.y = self.X_raw[y]

        elif self.labeled_size >= 20:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(pool.drop(y, axis=1), pool[y], random_state=42, test_size=0.2, stratify=pool[y])
        else:
            self.X_raw, self.y = self.unlabeled_pool.drop('target', axis=1), self.unlabeled_pool.target
            #raise Exception("Not enough labeled samples to fit classifier and generate test set")

    # utility function used to teach an active learner a new sample tweet from a json object {idx: i, hash: h, label: l}
    # after being learned, the tweet is removed from the sampling pool and the new model performance is recorded
    def teach(self, tweet: Dict[str, Union[int, str]]):
        # extract data from tweet obj
        idx = int(tweet['idx'])
        label = int(tweet['label'])
        if self.X_raw.empty: return
        text: str = self.X_raw.iloc[idx].tweet
        y_new: np.ndarray = np.array([label], dtype=int)

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