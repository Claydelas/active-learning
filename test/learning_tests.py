from sklearn.linear_model import LogisticRegression
import scipy.sparse as sp
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from gensim.models.doc2vec import Doc2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import os
import sys
import math
import pandas as pd
import numpy as np
import unittest
unittest.TestLoader.sortTestMethodsUsing = None

ROOT = os.path.abspath( os.path.dirname(  __file__ ) )
DATA_PATH = os.path.join( ROOT, '..', 'data' )

sys.path.insert(1, os.path.join(ROOT, '..', 'backend'))

from learning import Learning
from active_learning import ActiveLearning
import features
import preprocess as p


user_features = [('user_is_verified', 'bool'),
                 ('user_posts', 'numeric'),
                 ('user_likes', 'numeric'),
                 ('user_followers', 'numeric'),
                 ('user_friends', 'numeric')]
text_features = [('tweet', 'tweet')]
stats_features = [('emoji_count', 'numeric'),
                  ('polarity', 'numeric'),
                  ('subjectivity', 'numeric'),
                  ('hashtag_count', 'numeric'),
                  ('mentions_count', 'numeric'),
                  ('words_count', 'numeric'),
                  ('char_count', 'numeric'),
                  ('url_count', 'numeric'),
                  ('is_retweet', 'bool'),
                  ('tweet_likes', 'numeric'),
                  ('tweet_retweets', 'numeric'),
                  ('tweet_is_quote', 'bool')]


class TestLearning(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.doc2vec = Doc2Vec(vector_size=50, min_count=2, epochs=2)
        cls.tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
        cls.mock = pd.DataFrame(
            np.array([
                ['Sed vitae neque ac sapien imperdiet vestibulum. Sed euismod dictum sodales. Nullam dolor ligula, mattis non vulputate vel, fringilla lectus.', 5, 1, 0],
                ['Vivamus non mi eu diam gravida finibus vitae nec dui. Sed ac aliquam risus. Aenean convallis leo id nunc volutpat vulputate. Orci varius ex.', 100, 0, 1],
                ['Morbi nec facilisis lectus. Curabitur ac dolor sed magna venenatis euismod. Ut enim mauris, commodo sit amet posuere quis, pellentesque vel.', 1500, 0, 2]]),
            columns=['text_feature', 'numeric_feature', 'bool_feature', 'target'])
        p_path = os.path.join(DATA_PATH, "personal", "labeled_data.csv")
        cls.dataset = pd.read_csv(p_path, sep='\t')
        cls.learner = Learning(estimator=LogisticRegression(),
                               dataset=cls.dataset,
                               columns=text_features+user_features+stats_features,
                               vectorizer=cls.tfidf,
                               name='test+real',
                               targets=[{'val': 0, 'name': 'non-malicious'},
                                        {'val': 1, 'name': 'malicious'}])

    @classmethod
    def tearDownClass(cls):
        del cls.learner

    def test_1_preprocess(self):
        clean_df, clean_columns = p.transform(self.learner.dataset, self.learner.columns)
        extracted_features = ['emoji_count', 'polarity', 'subjectivity', 'hashtag_count',
                              'mentions_count', 'words_count', 'char_count', 'url_count', 'is_retweet']
        for c, t in self.learner.columns:
            if t == 'text' or t == 'tweet':
                self.assertIn(f'{c}_clean', [cl[0] for cl in clean_columns])
                self.assertIn(f'{c}_clean', clean_df.columns)

        for f in extracted_features:
            self.assertIn(f, clean_df)

    def test_2_learn_vectorizer(self):
        tfidf = self.tfidf
        with self.assertRaises(NotFittedError):
            check_is_fitted(tfidf, 'vocabulary_')
        features.learn_text_model(vectorizer=tfidf, dataset=self.learner.dataset, documents='tweet')
        self.assertIsNone(check_is_fitted(tfidf, 'vocabulary_'))

        doc2vec = self.doc2vec
        self.assertEqual(doc2vec.corpus_total_words, 0)
        train_count = doc2vec.train_count
        features.learn_text_model(vectorizer=doc2vec, dataset=self.learner.dataset, documents='tweet')
        self.assertGreater(doc2vec.corpus_total_words, 0)
        self.assertEqual(doc2vec.train_count, train_count+1)

    def test_3_split_ml(self):

        with self.assertRaises(Exception):
            self.learner.split(self.mock)

        dsize = len(self.learner.dataset.index)
        self.learner.split(self.learner.dataset, y='target', test_size=0.1)
        self.assertAlmostEqual(self.learner.labeled_size, dsize*0.9)
        self.assertAlmostEqual(self.learner.X_train.shape[0], dsize*0.9)
        self.assertEqual(self.learner.X_train.shape[0], self.learner.y_train.shape[0])
        self.assertAlmostEqual(self.learner.X_test.shape[0], dsize*0.1)
        self.assertEqual(self.learner.X_test.shape[0], self.learner.y_test.shape[0])

    def test_4_build_features(self):
        feature_cols = [('text_feature', 'text'),
                        ('numeric_feature', 'numeric'),
                        ('bool_feature', 'bool')]

        vectorD = self.tfidf.max_features
        feature_matrix = features.build_features(self.mock, feature_cols, self.tfidf)
        rows, cols = self.mock.shape[0], len(feature_cols) + vectorD - 1
        self.assertEqual(feature_matrix.shape[0], rows)
        self.assertEqual(feature_matrix.shape[1], cols)
        self.assertEqual(feature_matrix.format, 'csr')

        vectorD = self.doc2vec.vector_size
        feature_matrix = features.build_features(self.mock, feature_cols, self.doc2vec)
        rows, cols = self.mock.shape[0], len(feature_cols) + vectorD - 1
        self.assertEqual(feature_matrix.shape[0], rows)
        self.assertEqual(feature_matrix.shape[1], cols)
        self.assertFalse(sp.issparse(feature_matrix))

        with self.assertRaises(Exception):
            features.build_features(self.mock, [], self.tfidf)

    def test_5_fit(self):
        classifier = self.learner.estimator
        with self.assertRaises(NotFittedError):
            classifier.predict(self.learner.X_test)
        classifier.fit(self.learner.X_train, self.learner.y_train)
        self.assertIsNotNone(classifier.classes_)


class TestLearningAL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
        cls.mock = pd.DataFrame(
            np.array([
                ['Sed vitae neque ac sapien imperdiet vestibulum. Sed euismod dictum sodales. Nullam dolor ligula, mattis non vulputate vel, fringilla lectus.', 5, 1, 0],
                ['Vivamus non mi eu diam gravida finibus vitae nec dui. Sed ac aliquam risus. Aenean convallis leo id nunc volutpat vulputate. Orci varius ex.', 100, 0, 1],
                ['Morbi nec facilisis lectus. Curabitur ac dolor sed magna venenatis euismod. Ut enim mauris, commodo sit amet posuere quis, pellentesque vel.', 1500, 0, 2]]),
            columns=['text_feature', 'numeric_feature', 'bool_feature', 'target'])
        p_path = os.path.join(DATA_PATH, "processed", "personal_processed.pkl")
        cls.dataset = pd.read_pickle(p_path)
        cls.learner = ActiveLearning(estimator=LogisticRegression(),
                                     dataset=cls.dataset,
                                     columns=text_features+user_features+stats_features,
                                     vectorizer=cls.tfidf,
                                     learn_vectorizer=True,
                                     name='test+real+AL',
                                     targets=[{'val': 0, 'name': 'non-malicious'},
                                              {'val': 1, 'name': 'malicious'}])
        cls.learner.start(auto=False, server=False)

    @classmethod
    def tearDownClass(cls):
        del cls.learner

    def test_6_split_al(self):
        with self.assertRaises(Exception):
            self.learner.split(self.mock)

        dsize = len(self.learner.dataset.index)
        n_per_class = 1
        classes = len(np.unique(self.learner.dataset.target.values))
        trained_count = classes * n_per_class

        self.learner.split(self.learner.dataset, y='target',
                           test_size=0.1, n_per_class=n_per_class)

        self.assertEqual(self.learner.dataset_size, math.floor(dsize*0.9))

        self.assertEqual(self.learner.labeled_size, trained_count)

        self.assertEqual(self.learner.X_train.shape[0], trained_count)
        self.assertEqual(self.learner.X_train.shape[0], self.learner.y_train.shape[0])

        self.assertEqual(self.learner.X_test.shape[0], math.ceil(dsize*0.1))
        self.assertEqual(self.learner.X_test.shape[0], self.learner.y_test.shape[0])

        self.assertEqual(self.learner.X_pool.shape[0], math.floor(dsize*0.9)-trained_count)
        self.assertEqual(self.learner.X_pool.shape[0], self.learner.X_pool_df.shape[0])

    def test_7_teach(self):
        idx, sample = self.learner.learner.query(self.learner.X_pool)
        idx = int(idx)

        self.assertEqual(np.sum(self.learner.X_pool[idx] != sample), 0)

        label = random.randint(0, 1)
        labels = self.learner.labeled_size
        accuracies = len(self.learner.accuracy_scores)
        pool_size = self.learner.X_pool.shape[0]
        row = self.learner.X_pool_df.iloc[idx].name
        self.learner.teach({'idx': idx,
                            'label': label},
                           hashed=False)

        self.assertNotEqual(np.sum(self.learner.X_pool[idx] != sample), 0)
        self.assertEqual(self.learner.dataset.loc[row].target, label)
        self.assertEqual(self.learner.labeled_size, labels+1)

        with self.assertRaises(KeyError):
            self.learner.X_pool_df.loc[row]

        self.assertEqual(self.learner.X_pool.shape[0], pool_size-1)
        self.assertEqual(len(self.learner.accuracy_scores), accuracies+1)

    def test_8_skip(self):
        idx = random.randint(0, self.learner.X_pool.shape[0])
        sample = self.learner.X_pool[idx]

        dataset_size = self.learner.dataset_size
        pool_size = self.learner.X_pool.shape[0]

        row = self.learner.X_pool_df.iloc[idx].name
        self.learner.skip({'idx': idx},
                          hashed=False)

        self.assertNotEqual(np.sum(self.learner.X_pool[idx] != sample), 0)

        with self.assertRaises(KeyError):
            self.learner.X_pool_df.loc[row]

        self.assertEqual(self.learner.X_pool.shape[0], pool_size-1)
        self.assertEqual(self.learner.dataset_size, dataset_size-1)

    def test_9_auto_teach(self):
        for _ in range(0, 5):
            idx, sample = self.learner.learner.query(self.learner.X_pool)
            idx = int(idx)

            self.assertEqual(np.sum(self.learner.X_pool[idx] != sample), 0)

            labels = self.learner.labeled_size
            accuracies = len(self.learner.accuracy_scores)
            pool_size = self.learner.X_pool.shape[0]
            row = self.learner.X_pool_df.iloc[idx].name
            self.learner.auto_teach(1)

            self.assertNotEqual(np.sum(self.learner.X_pool[idx] != sample), 0)
            self.assertEqual(self.learner.labeled_size, labels+1)

            with self.assertRaises(KeyError):
                self.learner.X_pool_df.loc[row]

            self.assertEqual(self.learner.X_pool.shape[0], pool_size-1)
            self.assertEqual(len(self.learner.accuracy_scores), accuracies+1)

        labels = self.learner.labeled_size
        accuracies = len(self.learner.accuracy_scores)
        pool_size = self.learner.X_pool.shape[0]

        self.learner.auto_teach(5)

        self.assertEqual(self.learner.X_pool.shape[0], pool_size-5)
        self.assertEqual(self.learner.labeled_size, labels+5)
        self.assertEqual(len(self.learner.accuracy_scores), accuracies+5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
