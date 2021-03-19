import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from active_learning import ActiveLearning
import gensim
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Active Learning /w GUI
learner = ActiveLearning(KNeighborsClassifier(n_neighbors=1), 
                            dataset=pd.read_csv("data\merged.csv", sep='\t', index_col=0),
                            vectorizer = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
                        ).start()