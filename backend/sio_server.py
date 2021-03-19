from flask import Flask, request
import modAL.uncertainty
from flask_socketio import SocketIO
class Server():
    def __init__(self, learning):
        self.learning = learning
        self.app = Flask('Active Learning')
        self.sio = SocketIO(self.app, cors_allowed_origins='*')
        self.bootstrap()

    def run(self):
        # Start Web Server + Socket.IO
        # thread = threading.Thread(target=lambda: sio.run(app)).start()
        self.sio.run(self.app)

    def init(self, learning):
        # retrieve most uncertain instance
        query_idx, query_sample = learning.learner.query(learning.X)
        idx = int(query_idx)
        self.sio.emit('init', {
                'idx': idx,
                'text': learning.X_raw.iloc[idx].tweet,
                'uncertainty': modAL.uncertainty.classifier_uncertainty(classifier=learning.estimator, X=query_sample)[0],
                'series': learning.accuracy_scores,
                'strategy': 'uncertainty',
                'labeled_size': learning.labeled_size,
                'dataset_size': learning.dataset_size,
                'score': learning.accuracy_scores[-1]['score'] if learning.accuracy_scores else 0,
                'target': 80.00
                })
                
    def query(self, learning):
        # retrieve most uncertain instance
        query_idx, query_sample = learning.learner.query(learning.X)
        idx = int(query_idx)
        self.sio.emit('query', {
                'idx': idx,
                'text': learning.X_raw.iloc[idx].tweet,
                'uncertainty': modAL.uncertainty.classifier_uncertainty(classifier=learning.estimator, X=query_sample)[0],
                'labeled_size': learning.labeled_size,
                'series': learning.accuracy_scores[-1],
                'score': learning.accuracy_scores[-1]['score']
                })

    def bootstrap(self):
        @self.app.route('/')
        def index():
            return 'Server running on localhost:5000.'

        @self.sio.on('connect')
        def connect():
            #logging.info(f'Client connected: {request.sid}')
            self.init(self.learning)
            
        @self.sio.on('disconnect')
        def disconnect():
            pass
            #logging.info(f'Client disconnected: {request.sid}')

        @self.sio.on('refresh')
        def refresh():
            #logging.info(f'{request.sid} requested refresh.')
            self.init(self.learning)
            
        @self.sio.on('label')
        def label(tweet):
            self.learning.teach(tweet)
            self.query(self.learning)
    