import logging, sys, json, os
from typing import Callable, Collection, Dict, Tuple, Union

from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

import preprocess as pre

import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import gensim

from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from modAL.utils.data import data_hstack, retrieve_rows

Vectorizer = Union[TfidfVectorizer, gensim.models.doc2vec.Doc2Vec, gensim.models.word2vec.Word2Vec, gensim.models.keyedvectors.KeyedVectors]

# base class for learning processes
class Learning():
    def __init__(self,
                 estimator: BaseEstimator,
                 dataset: DataFrame,
                 columns: Collection[Tuple[str, str]],
                 vectorizer: Vectorizer = None,
                 learn_vectorizer: bool = False,
                 preprocess: bool = False,
                 extra_processing: Callable[[DataFrame], DataFrame] = None,
                 start: bool = False,
                 name: str = 'dataset',
                 targets = None):

        logging.basicConfig(handlers=[logging.FileHandler('server.log', 'a', 'utf-8')], level=logging.DEBUG, format='[%(asctime)s] %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

        self.estimator = estimator

        # default dataset
        if dataset is None or not isinstance(dataset, DataFrame):
            raise Exception("Dataset is not a valid pandas dataframe.")
        # append target column to dataframe if it doesn't exist
        if not 'target' in dataset.columns: dataset['target'] = np.nan

        if columns is None or len(columns) == 0:
            raise Exception("Please specify data attributes/features as a list of tuples [(column, type)].")
        
        # rename the attribute containing tweet text to "tweet"
        for col, type in columns:
            if type == 'tweet':
                dataset.rename(columns={col: 'tweet'}, inplace=True)

        # execute any aditional processing defined as a callback function (extra dataset-specific processing)
        if extra_processing is not None and callable(extra_processing): 
            dataset = extra_processing(dataset)
        self.dataset = dataset

        self.name = name
        self.targets = targets
        self.target_names = [l.get('name') for l in targets] if targets else None

        # rename the column containing tweet text to "tweet" (for feature extraction)
        self.columns = [('tweet', type) if type == 'tweet' else (col,type) for col, type in columns]
        self.accuracy_scores: Collection[Dict[str, Union[int, float]]] = []

        # default vectorizer
        if vectorizer is None:
            self.vectorizer = TfidfVectorizer(ngram_range=(1,3))
            self.learn_vectorizer = True
        else:
            self.vectorizer = vectorizer
            self.learn_vectorizer = learn_vectorizer

        self.preprocess = preprocess

        # start
        if start:
            self.start()


    def start(self):
        if self.preprocess: self.process()
        if self.learn_vectorizer: self.learn_text_model(vectorizer=self.vectorizer, dataset=self.dataset)
        self.split()
        self.fit(X=self.X_train, y=self.y_train)
        return self


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
        self.save(f"data/processed/{self.name}_processed.pkl")
        return self.dataset


    # trains a vectorizer on a set of documents
    @staticmethod
    def learn_text_model(vectorizer: Vectorizer, dataset: DataFrame, documents: str = 'tweet_clean') -> Vectorizer:
        if isinstance(vectorizer, TfidfVectorizer):
            # learn TF-IDF language model
            vectorizer.fit(dataset[documents])
        elif isinstance(vectorizer, gensim.models.doc2vec.Doc2Vec):
            # learn Doc2Vec language model
            train_corpus = [gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(row[documents]), [index]) for index, row in dataset.iterrows()]
            vectorizer.build_vocab(train_corpus)
            vectorizer.train(train_corpus, total_examples=vectorizer.corpus_count, epochs=vectorizer.epochs)
        return vectorizer


    def split(self, pool: DataFrame = None, y: str = 'target', test_size = 0.1):

        if pool is None: pool = self.dataset[self.dataset.target.notnull()]

        labels = len(pool.index)
        if labels >= 50:
            train, test = train_test_split(pool, random_state=42, test_size=test_size)

            self.X_train = self.build_features(train, self.columns)
            self.y_train = train[y]

            self.X_test = self.build_features(test, self.columns)
            self.y_test = test[y]
        else:
            raise Exception("Not enough labeled samples to fit classifier and generate test set")
        self.labeled_size = len(train.index)
        self.dataset_size = len(self.dataset.index) - len(test.index)


    # builds and stacks feature matrices to obtain a single matrix used for sampling and training
    def build_features(self, pool: DataFrame, columns: Collection[Tuple[str, str]]):
        blocks = []
        for column, type in columns:
            if type == 'text' or type == 'tweet':
                blocks.append(self._vectorize_(pool[column]))
            elif type == 'numeric':
                X = StandardScaler().fit_transform(pool[column].values.reshape(-1,1))
                blocks.append(X)
            elif type == 'bool':
                blocks.append(pool[column].apply(lambda val: 1 if val == True else 0).values.reshape(-1,1))
        return data_hstack(blocks)


    # utility function that provides a uniform method for vectorizing text via different vectorizers
    def _vectorize_(self, documents, vectorizer: Vectorizer = None):
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
        '''Produces the vector for a text based on the vectors of each individual word in @doc.'''
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

    @staticmethod
    def _get_row_(feature_matrix, idx: int):
        '''Utility function that returns row @idx from @feature_matrix as a 2d np array regardless of matrix format.''' 
        if isinstance(feature_matrix, np.ndarray):
            return retrieve_rows(feature_matrix, [idx])
        else:
            return retrieve_rows(feature_matrix, idx)


    def fit(self, X, y):
        self.estimator.fit(X=X, y=y)
        return self.estimator


    def f1_score(self, X, y, average = 'micro'):
        return f1_score(y_pred=self.estimator.predict(X), y_true=y, average=average)


    def classification_report(self, X, y, targets = None):
        return classification_report(y_pred=self.estimator.predict(X), y_true=y, target_names=targets, output_dict=True)


    def save(self, path:str):
        if not os.path.exists('data/processed'):
            os.makedirs('data/processed')
        self.dataset.to_pickle(path)

    def results(self, name: str ='', save:bool = False):
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

    @staticmethod
    def _eq_split_(X, y, n_per_class, random_state=None):
        if random_state:
            np.random.seed(random_state)
        sampled = X.groupby(y, sort=False).apply(
            lambda frame: frame.sample(n_per_class))
        mask = sampled.index.get_level_values(1)
    
        X_train = X.drop(mask)
        X_test = X.loc[mask]
    
        return X_train, X_test