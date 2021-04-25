# builds and stacks feature matrices to obtain a single matrix used for sampling and training
from typing import List, Tuple, Union

import gensim
import scipy.sparse as sp
import numpy as np
from modAL.utils.data import data_hstack
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

Vectorizer = Union[
    TfidfVectorizer, gensim.models.doc2vec.Doc2Vec, gensim.models.word2vec.Word2Vec,
    gensim.models.keyedvectors.KeyedVectors]


def build_features(pool: DataFrame, columns: List[Tuple[str, str]], vectorizer: Vectorizer):
    blocks = []
    for column, f_type in columns:
        if column in pool.columns:
            if f_type == 'text' or f_type == 'tweet':
                blocks.append(_vectorize_(pool[column], vectorizer))
            elif f_type == 'numeric':
                x = StandardScaler().fit_transform(pool[column].values.reshape(-1, 1))
                blocks.append(x)
            elif f_type == 'bool':
                blocks.append(pool[column].apply(lambda val: 1 if val else 0).values.reshape(-1, 1))
    if not blocks: raise Exception("Feature set can't be empty.")
    X = data_hstack(blocks)
    if sp.issparse(X):
        return X.tocsr()
    return X

    # utility function that provides a uniform method for vectorizing text via different vectorizers


def _vectorize_(documents, vectorizer: Vectorizer):
    if isinstance(vectorizer, TfidfVectorizer):
        return vectorizer.transform(documents)
    elif isinstance(vectorizer, gensim.models.doc2vec.Doc2Vec):
        return np.array([vectorizer.infer_vector(gensim.utils.simple_preprocess(x)) for x in documents])
    elif isinstance(vectorizer, gensim.models.keyedvectors.KeyedVectors):
        w_set = set(vectorizer.wv.index2word)
        return np.array([_avg_w2v_(vectorizer, x, w_set) for x in documents])


def _avg_w2v_(vectorizer, doc, w_set):
    """Produces the vector for a text based on the vectors of each individual word in @doc."""
    feature_vec = np.zeros((vectorizer.vector_size,), dtype="float32")
    n_words = 0
    words = doc.split()
    for word in words:
        if word in w_set:
            n_words = n_words + 1
            feature_vec = np.add(feature_vec, vectorizer[word])

    if n_words > 0:
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec


# trains a vectorizer on a set of documents
def learn_text_model(vectorizer: Vectorizer, dataset: DataFrame, documents: str = 'tweet_clean') -> Vectorizer:
    if isinstance(vectorizer, TfidfVectorizer):
        # learn TF-IDF language model
        vectorizer.fit(dataset[documents])
    elif isinstance(vectorizer, gensim.models.doc2vec.Doc2Vec):
        # learn Doc2Vec language model
        train_corpus = [
            gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(row[documents]), [index]) for
            index, row in dataset.iterrows()]
        vectorizer.build_vocab(train_corpus)
        vectorizer.train(train_corpus, total_examples=vectorizer.corpus_count, epochs=vectorizer.epochs)
    return vectorizer

