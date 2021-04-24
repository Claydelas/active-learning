import hashlib
from typing import Callable, Dict, Tuple, Union, List

import modAL.uncertainty
import numpy as np
import pandas as pd
from modAL.models import ActiveLearner
from modAL.utils.data import drop_rows
from pandas.core.frame import DataFrame
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from features import build_features
from learning import Learning, Vectorizer
from utils import get_row, eq_split


class ActiveLearning(Learning):
    def __init__(self,
                 estimator: BaseEstimator,
                 dataset: DataFrame,
                 columns: List[Tuple[str, str]],
                 vectorizer: Vectorizer = None,
                 learn_vectorizer: bool = False,
                 preprocess: bool = False,
                 extra_processing: Callable[[DataFrame], DataFrame] = None,
                 name: str = 'dataset',
                 targets=None,
                 query_strategy: Callable[..., np.ndarray] = modAL.uncertainty.uncertainty_sampling,
                 target_score: float = 80.00,
                 n_queries: int = None):
        super().__init__(estimator, dataset, columns, vectorizer, learn_vectorizer, preprocess, extra_processing,
                         name, targets)
        assert callable(query_strategy), 'query_strategy must be callable'
        self.query_strategy = query_strategy
        self.target_score = target_score
        self.n_queries = n_queries

        self.learner = ActiveLearner(
            estimator=self.estimator,
            query_strategy=self.query_strategy
        )

    # utility function that provides a shortcut for building the active learning workflow
    def start(self, auto: bool = False, server: bool = True):
        super().start()
        if auto:
            self.auto_teach(self.n_queries)
        if server:
            self.start_server()

    def split(self, pool: DataFrame = None, y: str = 'target', test_size=0.1, n_per_class=1):

        if pool is None:
            pool = self.dataset[self.dataset.target.notnull()]
        unlabeled_frame = self.dataset[self.dataset.target.isnull()]

        labels = len(pool.index)
        if labels >= 50:
            pool_train, test = train_test_split(pool, random_state=42, test_size=test_size)
            self.X_test = build_features(test, self.columns, self.vectorizer)
            self.y_test = test[y]

            unlabeled_pool, train = eq_split(pool_train, pool_train[y], n_per_class=n_per_class, random_state=42)

            self.X_train = build_features(train, self.columns, self.vectorizer)
            self.y_train = train[y].to_numpy()

            # merge with previously unlabeled instances if any
            X_pool = pd.concat([unlabeled_pool, unlabeled_frame])

            self.X_pool_df = X_pool
            self.X_pool = build_features(X_pool, self.columns, self.vectorizer)

            self.labeled_size = len(train.index)
            self.dataset_size = len(self.dataset.index) - len(test.index)
        else:
            self.X_pool_df = unlabeled_frame
            self.X_pool = build_features(unlabeled_frame, self.columns, self.vectorizer)

            self.labeled_size = labels
            self.dataset_size = len(self.dataset.index)
            # raise Exception("Not enough labeled samples to fit classifier and generate test set")

    def auto_teach(self, queries=None):
        if queries and queries > self.X_pool.shape[0]:
            queries = self.X_pool.shape[0]
        for _ in range(0, queries) if queries else range(0, self.X_pool.shape[0]):
            # retrieve most uncertain instance
            idx, sample = self.learner.query(self.X_pool)
            idx = int(idx)
            dataset_idx = self.X_pool_df.iloc[idx].name
            self.learner.teach(sample, np.array([self.dataset.loc[dataset_idx].target], dtype=int))
            self.labeled_size += 1
            # remove learned sample from pool
            self.X_pool = drop_rows(self.X_pool, idx)
            self.X_pool_df = self.X_pool_df.drop(dataset_idx)
            # store accuracy metric after training
            self.accuracy_scores.append(
                dict(self.classification_report(self.X_test, self.y_test, self.target_names), labels=self.labeled_size))

    # utility function used to teach an active learner a new sample tweet from a json object {idx: i, hash: h, label: l}
    # after being learned, the tweet is removed from the sampling pool and the new model performance is recorded
    def teach(self, tweet: Dict[str, Union[int, str]], hashed=False):
        # extract data from tweet obj
        idx = int(tweet['idx'])
        label = int(tweet['label'])

        if self.X_pool.shape[0] == 0:
            return False

        row = self.X_pool_df.iloc[idx]
        text: str = row.tweet
        # fail early if hashes don't match (web app out of sync)
        if hashed and tweet['hash'] != hashlib.md5(text.encode()).hexdigest():
            return False

        dataset_idx = row.name
        y_new: np.ndarray = np.array([label], dtype=int)

        # teach new sample
        self.learner.teach(get_row(self.X_pool, idx), y_new)
        self.dataset.at[dataset_idx, 'target'] = label
        self.labeled_size += 1

        # remove learned sample from pool
        self.X_pool = drop_rows(self.X_pool, idx)
        self.X_pool_df = self.X_pool_df.drop(dataset_idx)

        # store accuracy metric after training
        self.accuracy_scores.append(
            dict(self.classification_report(self.X_test, self.y_test, self.target_names), labels=self.labeled_size))
        return True

    def skip(self, tweet: Dict[str, Union[int, str]], hashed=False):
        # extract data from tweet obj
        idx = int(tweet['idx'])
        if self.X_pool.shape[0] == 0:
            return False

        row = self.X_pool_df.iloc[idx]
        text: str = row.tweet
        # fail early if hashes don't match (web app out of sync)
        if hashed and tweet['hash'] != hashlib.md5(text.encode()).hexdigest():
            return False

        dataset_idx = row.name

        # skip sample
        self.X_pool = drop_rows(self.X_pool, idx)
        self.X_pool_df = self.X_pool_df.drop(dataset_idx)
        self.dataset_size -= 1
        return True

    # starts an instance of the backend server used by the labeling web-app
    def start_server(self):
        from sio_server import Server
        server = Server(self)
        server.run()

    def fit(self, X, y):
        self.learner.fit(X=X, y=y)
        return self.learner
