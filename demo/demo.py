#%% ----------------------------------------------------------------imports
import sys, os, json
ROOT = os.path.abspath( os.path.dirname(  __file__ ) )
DATA_PATH = os.path.join( ROOT, '..', 'data' )

sys.path.insert(1, os.path.join(ROOT, '..', 'backend'))

from active_learning import ActiveLearning
from learning import Learning
import features

from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import modAL

from sklearn.linear_model import LogisticRegression
#%% ----------------------------------------------------------------dataset
p_path = os.path.join(DATA_PATH, "processed", "personal_processed.pkl")
personal = pd.read_pickle(p_path)
#%% ----------------------------------------------------------------tfidf
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=8000,
    min_df=1,
    max_df=0.75
    )
vectorizer = features.learn_text_model(vectorizer, personal)
#%% ----------------------------------------------------------------
baseline = Learning(LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42),
                    dataset=personal,
                    columns=[('tweet', 'tweet')],
                    vectorizer=vectorizer,
                    name='personal-ML')
baseline.start()
results = baseline.results()[-1]
target = results['macro avg']['f1-score']*100
print(json.dumps(results, indent=4))
#%% ----------------------------------------------------------------
active = ActiveLearning(LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42),
                    dataset=personal,
                    columns=[('tweet', 'tweet')],
                    query_strategy=modAL.uncertainty.uncertainty_sampling,
                    vectorizer=vectorizer,
                    target_score=target,
                    n_queries=100,
                    name='personal-AL')
active.start(auto=True, server=False)
results_al = active.results()
print(json.dumps(results_al, indent=4))
