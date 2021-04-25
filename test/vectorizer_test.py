import sys, os
ROOT = os.path.abspath( os.path.dirname(  __file__ ) )
DATA_PATH = os.path.join( ROOT, '..', 'data' )

sys.path.insert(1, os.path.join(ROOT, '..', 'backend'))

import numpy as np
import pandas as pd
import gensim.downloader as api
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from active_learning import ActiveLearning

p_path = os.path.join(DATA_PATH, "processed", "personal_processed.pkl")
personal = pd.read_pickle(p_path)
# %%
glove = api.load('glove-twitter-50')
glove_sim = ActiveLearning(estimator=LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'),
    dataset=personal,
    columns=[('tweet','tweet')],
    vectorizer=glove,
    name='cosine',
    targets=[{'val': 0,'name': 'non-malicious'},
             {'val': 1,'name': 'malicious'}],
    query_strategy=uncertainty_sampling)
glove_sim.start(auto=False, server=False)
# %%
top10_uncertainty = uncertainty_sampling(glove_sim.learner.estimator, glove_sim.X_pool, n_instances=10)
top10_margin = margin_sampling(glove_sim.learner.estimator, glove_sim.X_pool, n_instances=10)
top10_entropy = entropy_sampling(glove_sim.learner.estimator, glove_sim.X_pool, n_instances=10)
# %%
sim = cosine_similarity(glove_sim.learner.X_training, glove_sim.X_pool)[1]
sort = np.argsort(sim)
indices = sort[:20]
vectors = sim[indices]
tweet_vectors = glove_sim.X_pool[indices]
tweets_cosine = glove_sim.X_pool_df.iloc[indices].tweet
idx_sampled = uncertainty_sampling(glove_sim.learner.estimator, glove_sim.X_pool, n_instances=20)
tweets_sampled = glove_sim.X_pool_df.iloc[idx_sampled].tweet
print(indices)
print(idx_sampled)
intersection = np.intersect1d(indices, idx_sampled)
print(intersection)
print("GloVe and Uncertainty Sampling agree on", len(intersection), "samples.")