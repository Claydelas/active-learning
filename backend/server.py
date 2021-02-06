import logging
import threading
from datetime import datetime
from time import sleep

from flask import Flask, request
from flask_socketio import SocketIO

logging.basicConfig(filename='server.log', level=logging.DEBUG, format='[%(asctime)s] %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

app = Flask('Active Learning')
sio = SocketIO(app, cors_allowed_origins='*')


@app.route('/')
def index():
    return 'Server running on localhost:5000.'


# Start Web Server + Socket.IO
thread = threading.Thread(target=lambda: sio.run(app)).start()


@sio.on('connect')
def connect():
    logging.info(f'Client connected: {request.sid}')
    # TODO: on connect emit data samples for labeling


@sio.on('disconnect')
def disconnect():
    logging.info(f'Client disconnected: {request.sid}')


@sio.on('refresh')
def refresh():
    # TODO: Refresh AL process
    print('refresh')


@sio.on('label')
def label(tweet):
    # TODO: Teach new label
    print(f'idx {tweet.idx}, label {tweet.label}')


# normal script execution below:
print('Start AL process')

# sample real-time clock for testing
while True:
    sio.emit('tweet', {'idx': 0, 'text': datetime.now().strftime("%H:%M:%S")})
    sleep(1)
