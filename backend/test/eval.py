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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#%% ----------------------------------------------------------------datasets
#t_davidson = pd.read_pickle("../../data/processed/t-davidson_processed.pkl")
#t_davidson = pd.read_pickle("data/processed/t-davidson_processed.pkl")
personal = pd.read_pickle("../../data/processed/personal_processed.pkl")
#%% ----------------------------------------------------------------word embeddings
glove = api.load('glove-twitter-50')
#%% ----------------------------------------------------------------doc2vec
doc2vec = Doc2Vec(vector_size=50, min_count=2, epochs=20)
#%% ----------------------------------------------------------------tfidf2
tfidf = TfidfVectorizer(
    ngram_range=(1, 3),
    decode_error='replace',
    max_features=5000,
    min_df=5,
    max_df=0.75
    )
# %%
text_features = [('tweet', 'tweet')]
user_features = [('user_is_verified','bool'),
                ('user_posts', 'numeric'),
                ('user_likes','numeric'),
                ('user_followers', 'numeric'),
                ('user_friends', 'numeric')]
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
options = {
    'classifiers': [
        {'name': 'Logistic Regression', 'classifier': LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)},
        {'name': 'Linear SVM', 'classifier': SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)},
        {'name': 'Random Forest', 'classifier': RandomForestClassifier(class_weight='balanced', max_depth=8, n_estimators=200, random_state=42, n_jobs= -1)},
        {'name': 'KNN', 'classifier': KNeighborsClassifier(n_neighbors=1, n_jobs= -1)}],
    'datasets': [
        {'name': 'personal',
         'df': personal,
         'targets': [
             {'val': 0,'name': 'non-malicious'},
             {'val': 1,'name': 'malicious'}]
        }],
    'vectorizers': [
        {'name': 'TF IDF Vectorizer', 'vectorizer': tfidf},
        {'name': 'GloVe Word Embeddings', 'vectorizer': glove},
        {'name': 'Doc2Vec Paragraph Embeddings', 'vectorizer': doc2vec}],
    'features': [
        {'name': 'text', 'cols': text_features},
        {'name': 'text+stats', 'cols': text_features + stats_features},
        {'name': 'text+user', 'cols': text_features + user_features},
        {'name': 'text+user+stats', 'cols': text_features + user_features + stats_features}],
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
    transformer, dataset = None, None
    for v, d in transformers:
        if type(v) == type(p['vectorizers']['vectorizer']) and p['datasets']['name'] == d['name']:
            transformer, dataset = v, d
            break
    if transformer is None or dataset is None: 
        transformer, dataset = p['vectorizers']['vectorizer'], p['datasets']
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
                  name=al_name,
                  targets=targets
                  )
    al.start(auto=True, server=False)
    al_results = al.results(save=True)[-1]
    eval_scores.append(dict(al_results, name=al_name))
    del al
print(json.dumps(eval_scores, indent=4))
with open(f'results/final/eval_results.json', 'w', encoding='utf-8') as f:
    json.dump(eval_scores, f, ensure_ascii=False, indent=4)

# %%
