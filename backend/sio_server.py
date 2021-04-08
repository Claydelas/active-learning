from logging import Logger
from flask import Flask, request
import modAL.uncertainty
from flask_socketio import SocketIO


class Server():
    def __init__(self, learning, logging: Logger):
        self.learning = learning
        self.logging = logging
        self.app = Flask('Active Learning')
        self.sio = SocketIO(self.app, cors_allowed_origins='*')
        self.bootstrap()

    def run(self):
        # Start Web Server + Socket.IO
        # thread = threading.Thread(target=lambda: self.sio.run(self.app)).start()
        self.sio.run(self.app)

    def init(self, learning):
        # retrieve most uncertain instance
        if learning.X_pool.shape[0] > 0:
            query_idx, query_sample = learning.learner.query(learning.X_pool)
            idx = int(query_idx)
            self.sio.emit('init', {
                'idx': idx,
                'text': learning.X_pool_raw.iloc[idx].tweet,
                'uncertainty': modAL.uncertainty.classifier_uncertainty(classifier=learning.estimator, X=query_sample)[0],
                'series': learning.accuracy_scores,
                'labeled_size': learning.labeled_size,
                'dataset_size': learning.dataset_size,
                'score': learning.accuracy_scores[-1]['macro avg']['f1-score'] if learning.accuracy_scores else 0,
                'target': learning.target_score,
                'report': learning.accuracy_scores[-1] if learning.accuracy_scores else {}
            })
        else:
            self.sio.emit('end', {
                'series': learning.accuracy_scores,
                'labeled_size': learning.labeled_size,
                'dataset_size': learning.dataset_size,
                'score': learning.accuracy_scores[-1]['macro avg']['f1-score'] if learning.accuracy_scores else 0,
                'target': learning.target_score,
                'report': learning.accuracy_scores[-1] if learning.accuracy_scores else {}
            })

    def query(self, learning):
        # retrieve most uncertain instance
        if learning.X_pool.shape[0] > 0:
            query_idx, query_sample = learning.learner.query(learning.X_pool)
            idx = int(query_idx)
            self.sio.emit('query', {
                'idx': idx,
                'text': learning.X_pool_raw.iloc[idx].tweet,
                'uncertainty': modAL.uncertainty.classifier_uncertainty(classifier=learning.estimator, X=query_sample)[0],
                'labeled_size': learning.labeled_size,
                'series': learning.accuracy_scores[-1],
                'score': learning.accuracy_scores[-1]['macro avg']['f1-score'],
                'report': learning.accuracy_scores[-1]
            })
        else:
            self.sio.emit('end', {
                'labeled_size': learning.labeled_size,
                'series': learning.accuracy_scores,
                'score': learning.accuracy_scores[-1]['macro avg']['f1-score'],
                'report': learning.accuracy_scores[-1]
            })

    def bootstrap(self):
        @self.app.route('/')
        def index():
            return 'Server running on localhost:5000.'

        @self.sio.on('connect')
        def connect():
            self.logging.info(f'Client connected: {request.sid}')
            self.init(self.learning)

        @self.sio.on('disconnect')
        def disconnect():
            self.logging.info(f'Client disconnected: {request.sid}')

        @self.sio.on('refresh')
        def refresh():
            self.logging.info(f'{request.sid} requested refresh.')
            self.init(self.learning)

        @self.sio.on('label')
        def label(tweet):
            self.learning.teach(tweet, hashed=True)
            self.query(self.learning)

        @self.sio.on('checkpoint')
        def checkpoint():
            path = f"data/{self.learning.name}_cp.pkl"
            self.learning.save(path)
            return f"Dataset saved @{path}"
