import logging
import sys
import time
import re
from copy import deepcopy
from logging import Logger

import modAL.uncertainty
from flask import Flask, request
from flask_socketio import SocketIO

from active_learning import ActiveLearning


class Server:
    def __init__(self, learning: ActiveLearning = None, options=None, logger: Logger = None):
        if options is None:
            options = {}
        self.learning = learning
        self.options = options
        if logger is None:
            logging.basicConfig(handlers=[logging.FileHandler('server.log', 'a', 'utf-8')], level=logging.DEBUG, format='[%(asctime)s] %(message)s',
                                datefmt='%m/%d/%Y %I:%M:%S %p')
            logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
            self.logger = logging.getLogger()
        else:
            self.logger = logger
        self.app = Flask('Active Learning')
        self.sio = SocketIO(self.app, cors_allowed_origins='*')
        self.bootstrap()

    def run(self):
        # Start Web Server + Socket.IO
        # thread = threading.Thread(target=lambda: self.sio.run(self.app)).start()
        self.sio.run(self.app, port=5000, host='0.0.0.0')

    def parse_options(self, options):
        classifier = next(filter(lambda c: c['name'] == options['classifier'], self.options['classifiers']), {}).get('classifier')
        dataset = next(filter(lambda d: d['name'] == options['dataset'], self.options['datasets']), {})
        vectorizer = next(filter(lambda v: v['name'] == options['vectorizer'], self.options['vectorizers']), {}).get('vectorizer')
        query_strategy = next(filter(lambda q: q['name'] == options['query_strategy'], self.options['query_strategies']), {}).get('strategy')
        features = sum(filter(None, [next(filter(lambda f: f['name'] == key, self.options['features']), {}).get('cols') if val else [] for key, val in options['features'].items()]), [])
        if not features: return "No features were selected."
        if classifier is None: return "Selected classifier isn't supported."
        if dataset is None: return "Dataset doesn't exist."
        if vectorizer is None: return "Selected vectorizer isn't supported."
        if query_strategy is None: return "Query strategy isn't supported."
        dataset_df = dataset.get('df')
        if not bool(set([f[0] for f in features]).intersection(dataset.get('df').columns)): return "Dataset doesn't seem to contain any of the selected features."
        self.learning = ActiveLearning(estimator=classifier,
                                 dataset=dataset_df,
                                 columns=features,
                                 vectorizer=deepcopy(vectorizer),
                                 learn_vectorizer=True,
                                 query_strategy=query_strategy,
                                 targets=dataset.get('targets'),
                                 target_score=options.get('target'),
                                 name=f"{dataset.get('name')}-{classifier.__class__.__name__}-{vectorizer.__class__.__name__}-{query_strategy.__name__}-AL")
        self.learning.start(server=False)
        return "Success."

    def init(self, learning):
        if learning is None:
            self.sio.emit('options', {
                'classifiers': [c['name'] for c in self.options['classifiers']],
                'datasets': [d['name'] for d in self.options['datasets']],
                'vectorizers': [v['name'] for v in self.options['vectorizers']],
                'query_strategies': [q['name'] for q in self.options['query_strategies']],
                'features': [f['name'] for f in self.options['features']]
            })
        # retrieve most uncertain instance
        elif learning.X_pool.shape[0] > 0:
            query_idx, query_sample = learning.learner.query(learning.X_pool)
            idx = int(query_idx)
            self.sio.emit('init', {
                'idx': idx,
                'text': learning.X_pool_df.iloc[idx].tweet,
                'targets': learning.targets,
                'uncertainty': modAL.uncertainty.classifier_uncertainty(classifier=learning.estimator, X=query_sample)[0],
                'series': learning.accuracy_scores,
                'labeled_size': learning.labeled_size,
                'dataset_size': learning.dataset_size,
                'score': learning.accuracy_scores[-1]['macro avg']['f1-score'] if learning.accuracy_scores else 0,
                'target': learning.target_score
            })
        else:
            self.sio.emit('end', {
                'series': learning.accuracy_scores,
                'labeled_size': learning.labeled_size,
                'dataset_size': learning.dataset_size,
                'score': learning.accuracy_scores[-1]['macro avg']['f1-score'] if learning.accuracy_scores else 0,
                'target': learning.target_score
            })

    def query(self, learning):
        # retrieve most uncertain instance
        if learning.X_pool.shape[0] > 0:
            query_idx, query_sample = learning.learner.query(learning.X_pool)
            idx = int(query_idx)
            self.sio.emit('query', {
                'idx': idx,
                'text': learning.X_pool_df.iloc[idx].tweet,
                'uncertainty': modAL.uncertainty.classifier_uncertainty(classifier=learning.estimator, X=query_sample)[0],
                'labeled_size': learning.labeled_size,
                'dataset_size': learning.dataset_size,
                'series': learning.accuracy_scores,
                'score': learning.accuracy_scores[-1]['macro avg']['f1-score'] if learning.accuracy_scores else 0
            })
        else:
            self.sio.emit('end', {
                'labeled_size': learning.labeled_size,
                'series': learning.accuracy_scores,
                'score': learning.accuracy_scores[-1]['macro avg']['f1-score'] if learning.accuracy_scores else 0
            })

    def bootstrap(self):
        @self.app.route('/')
        def index():
            return 'Server running on localhost:5000.'

        @self.sio.on('connect')
        def connect():
            self.logger.info(f'Client connected: {request.sid}')
            self.init(self.learning)

        @self.sio.on('disconnect')
        def disconnect():
            self.logger.info(f'Client disconnected: {request.sid}')

        @self.sio.on('refresh')
        def refresh():
            self.logger.info(f'{request.sid} requested refresh.')
            self.init(self.learning)

        @self.sio.on('options')
        def build_model(options):
            success = False
            if self.learning is None:
                success = self.parse_options(options)
                self.init(self.learning)
            return success or "Model is already initialised."

        @self.sio.on('label')
        def label(tweet):
            if self.learning is None: return "Model is not initialised."
            success = self.learning.teach(tweet, hashed=True)
            if success:
                self.query(self.learning)
            return True

        @self.sio.on('skip')
        def skip(tweet):
            if self.learning is None: return "Model is not initialised."
            success = self.learning.skip(tweet, hashed=True)
            if success:
                self.query(self.learning)

        @self.sio.on('checkpoint')
        def checkpoint():
            if self.learning is None: return "Model is not initialised."
            path = f"{re.sub(r'/', ' ' , self.learning.name)}-{int(time.time())}.pkl"
            self.learning.save(path)
            return f"Dataset saved as {path}."

        @self.sio.on('reset')
        def reset():
            del self.learning
            self.learning = None
            self.init(None)
            return True
