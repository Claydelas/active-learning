#%% ----------------------------------------------------------------imports
import sys, os, json
from copy import deepcopy
from sklearn import clone
import pandas as pd
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from active_learning import ActiveLearning
from learning import Learning

import gensim.downloader as api
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.doc2vec import Doc2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling

from sklearn.model_selection import ParameterGrid

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
#%% ----------------------------------------------------------------datasets
t_davidson = pd.read_pickle("../../data/processed/t-davidson_processed.pkl")
#t_davidson = pd.read_pickle("data/processed/t-davidson_processed.pkl")
#%% ----------------------------------------------------------------word embeddings
#glove = api.load('glove-twitter-50')
#%% ----------------------------------------------------------------doc2vec
doc2vec = Doc2Vec(vector_size=50, min_count=2, epochs=20)
#%% ----------------------------------------------------------------tfidf2
tfidf = TfidfVectorizer(
    ngram_range=(1, 3),
    use_idf=True,
    smooth_idf=False,
    norm=None,
    decode_error='replace',
    max_features=5000,
    min_df=5,
    max_df=0.75
    )
# %%
textual_features = [('tweet', 'tweet')]
user_features = [('polarity', 'numeric')]
options = {
    'classifiers': [
        {'name': 'Logistic Regression', 'classifier': LogisticRegression(class_weight='balanced', penalty='l2', max_iter=500, solver='liblinear')}],
    'datasets': [
        {'name': 't_davidson',
         'df': t_davidson,
         'targets': [
             {'val': 0,'name': 'hate speech'},
             {'val': 1,'name': 'offensive language'},
             {'val': 2,'name': 'neither'}]
        }],
    'vectorizers': [
        {'name': 'TF IDF Vectorizer', 'vectorizer': tfidf}],
    'features': [
        {'name': 'text', 'cols': textual_features},
        {'name': 'text+user', 'cols': textual_features + user_features}],
    'query_strategies': [
        {'name': 'Uncertainty Sampling', 'strategy': uncertainty_sampling},
        {'name': 'Entropy Sampling', 'strategy': entropy_sampling},
        {'name': 'Margin Sampling', 'strategy': margin_sampling}],
    }
permutations = list(ParameterGrid(options))

transformers = []

for v in options.get('vectorizers'):
    for d in options.get('datasets'):
        if isinstance(v['vectorizer'], KeyedVectors):
            transformers.append((v['vectorizer'],d))
        else:
            learned_v = Learning.learn_text_model(deepcopy(v['vectorizer']), d['df'])
            transformers.append((learned_v,d))

eval_scores = []
seen = []

for p in permutations:

    pred = lambda x_y:(x:=x_y[0], y:=x_y[1], type(p['vectorizers']['vectorizer'] == type(x) and p['datasets'].equals(y)))

    transformer, dataset = next(filter(pred, transformers), (p['vectorizers']['vectorizer'], p['datasets']))
    estimator = clone(p['classifiers']['classifier'])
    features = p.get('features')
    strategy = p.get('query_strategies')
    targets = dataset['targets']

    ml_name = f"{dataset['name']}-{p['classifiers']['name']}-{p['vectorizers']['name']}-{features['name']}-ML"
    if ml_name not in seen:
        ml = Learning(estimator=estimator,
                      dataset=dataset['df'],
                      columns=features['cols'],
                      vectorizer=transformer,
                      name=ml_name,
                      targets=targets
                      )
        ml.start()
        ml_results = ml.results(save=True)[-1]
        eval_scores.append(dict(ml_results, name=ml_name))
        seen.append(ml_name)
        del ml

    al_name = f"{dataset['name']}-{p['classifiers']['name']}-{p['vectorizers']['name']}-{features['name']}-{strategy['name']}-AL"
    al = ActiveLearning(estimator=estimator,
                  dataset=dataset['df'],
                  columns=features['cols'],
                  vectorizer=transformer,
                  query_strategy=strategy['strategy'],
                  n_queries=1000,
                  name=al_name,
                  targets=targets
                  )
    al.start(auto=True, server=False)
    al_results = al.results(save=True)[-1]
    eval_scores.append(dict(al_results, name=al_name))
    del al
print(json.dumps(eval_scores, indent=4))
with open(f'results/results.json', 'w', encoding='utf-8') as f:
    json.dump(eval_scores, f, ensure_ascii=False, indent=4)
