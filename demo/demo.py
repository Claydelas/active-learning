#%% ----------------------------------------------------------------imports
import sys, os, json
sys.path.insert(1, os.path.join(sys.path[0], '../backend'))

from active_learning import ActiveLearning
from learning import Learning

from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import modAL

from sklearn.linear_model import LogisticRegression
#%% ----------------------------------------------------------------dataset
t_davidson = pd.read_pickle("data/processed/t-davidson_processed.pkl")
#%% ----------------------------------------------------------------tfidf
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    use_idf=True,
    smooth_idf=False,
    norm=None,
    decode_error='replace',
    max_features=10000,
    min_df=5,
    max_df=0.75
    )
vectorizer = Learning.learn_text_model(vectorizer, t_davidson)
#%% ----------------------------------------------------------------
baseline = Learning(LogisticRegression(class_weight='balanced', penalty='l2', max_iter=1000, C=0.01),
                    dataset=t_davidson,
                    columns=[('tweet', 'tweet')],
                    vectorizer=vectorizer,
                    name='t-davidson-ML')
baseline.start()
results = baseline.results()[-1]
target = results['macro avg']['f1-score']*100
print(json.dumps(results, indent=4))
#%% ----------------------------------------------------------------
active = ActiveLearning(LogisticRegression(class_weight='balanced', penalty='l2', max_iter=1000, C=0.01),
                    dataset=t_davidson,
                    columns=[('tweet', 'tweet')],
                    query_strategy=modAL.uncertainty.uncertainty_sampling,
                    vectorizer=vectorizer,
                    target_score=target,
                    name='t-davidson-AL')
active.start()
