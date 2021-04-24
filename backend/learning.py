import json
import os
from typing import Callable, Dict, Tuple, Union, List

import numpy as np
from pandas.core.frame import DataFrame
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split

import preprocess as p
# base class for learning processes
from features import Vectorizer, learn_text_model, build_features


class Learning:
    def __init__(self,
                 estimator: BaseEstimator,
                 dataset: DataFrame,
                 columns: List[Tuple[str, str]],
                 vectorizer: Vectorizer = None,
                 learn_vectorizer: bool = False,
                 preprocess: bool = False,
                 extra_processing: Callable[[DataFrame], DataFrame] = None,
                 name: str = 'dataset',
                 targets=None):

        self.estimator = estimator

        # default dataset
        if dataset is None or not isinstance(dataset, DataFrame):
            raise Exception("Dataset is not a valid pandas dataframe.")
        # append target column to dataframe if it doesn't exist
        if 'target' not in dataset.columns:
            dataset['target'] = np.nan

        if columns is None or len(columns) == 0:
            raise Exception("Please specify data attributes/features as a list of tuples [(column, type)].")

        # rename the attribute containing tweet text to "tweet"
        for col, f_type in columns:
            if f_type == 'tweet':
                dataset.rename(columns={col: 'tweet'}, inplace=True)

        # execute any additional processing defined as a callback function (extra dataset-specific processing)
        if extra_processing is not None and callable(extra_processing):
            dataset = extra_processing(dataset)
        self.dataset = dataset

        self.name = name
        self.targets = targets
        self.target_names = [target.get('name') for target in targets] if targets else None

        # rename the column containing tweet text to "tweet" (for feature extraction)
        self.columns = [('tweet', f_type) if f_type == 'tweet' else (col, f_type) for col, f_type in columns]
        self.accuracy_scores: List[Dict[str, Union[int, float]]] = []

        # default vectorizer
        if vectorizer is None:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 3))
            self.learn_vectorizer = True
        else:
            self.vectorizer = vectorizer
            self.learn_vectorizer = learn_vectorizer

        self.preprocess = preprocess

    def start(self):
        if self.preprocess:
            self.dataset, self.columns = p.transform(self.dataset, self.columns)
        if self.learn_vectorizer:
            learn_text_model(vectorizer=self.vectorizer, dataset=self.dataset)
        self.split()
        self.fit(X=self.X_train, y=self.y_train)
        self.accuracy_scores.append(
            dict(self.classification_report(self.X_test, self.y_test, self.target_names), labels=self.labeled_size))
        return self

    def split(self, pool: DataFrame = None, y: str = 'target', test_size=0.1):

        if pool is None:
            pool = self.dataset[self.dataset.target.notnull()]

        labels = len(pool.index)
        if labels >= 50:
            train, test = train_test_split(pool, random_state=42, test_size=test_size)

            self.X_train = build_features(train, self.columns, self.vectorizer)
            self.y_train = train[y]

            self.X_test = build_features(test, self.columns, self.vectorizer)
            self.y_test = test[y]
        else:
            raise Exception("Not enough labeled samples to fit classifier and generate test set")
        self.labeled_size = len(train.index)
        self.dataset_size = len(self.dataset.index) - len(test.index)

    def fit(self, X, y):
        self.estimator.fit(X=X, y=y)
        return self.estimator

    def f1_score(self, X, y, average='macro'):
        return f1_score(y_pred=self.estimator.predict(X), y_true=y, average=average)

    def classification_report(self, X, y, targets=None):
        return classification_report(y_pred=self.estimator.predict(X), y_true=y, target_names=targets, output_dict=True)

    def save(self, path: str):
        if not os.path.exists('data/processed'):
            os.makedirs('data/processed')
        self.dataset.to_pickle(path)

    def results(self, name: str = '', save: bool = False):
        if save:
            if not os.path.exists('results'):
                os.makedirs('results')
            with open(f'results/{name or self.name}.json', 'w', encoding='utf-8') as f:
                if len(self.accuracy_scores) > 0:
                    json.dump(self.accuracy_scores, f, ensure_ascii=False, indent=4)
                    return self.accuracy_scores
                else:
                    results = [self.classification_report(self.X_test, self.y_test, self.target_names)]
                    json.dump(results, f, ensure_ascii=False, indent=4)
                    return results

        if len(self.accuracy_scores) > 0:
            return self.accuracy_scores
        return [self.classification_report(self.X_test, self.y_test, self.target_names)]
