import { useState, useEffect } from 'react';
import './App.css';

const io = require("socket.io-client");
const socket = io('http://127.0.0.1:5000', {transports: ['websocket']});

function alright(tweet) {
  console.log(`"${tweet.text}" --> alright. --> 0`)
  socket.emit("label", {'idx': tweet.idx, 'label': 0})
}

function malicious(tweet) {
  console.log(`"${tweet.text}" --> malicious. --> 1`)
  socket.emit("label", {'idx': tweet.idx, 'label': 1})
}

function refresh() {
  console.log('refresh')
  socket.emit("refresh")
}

function App() {

  const [tweet, setTweet] = useState({idx: -1, text: "Tweet text will be displayed here"});
  const [uncertainty, setUncertainty] = useState(0.00);
  const [progress, setProgress] = useState(0);
  const [total, setTotal] = useState(0);
  const [score, setScore] = useState(0.00);

  useEffect(() => {
    socket.on("tweet", data => {
      setTweet(data)
    });
  });

  return (
    <div className="App">
      <div className="App-main">
        <div className="tweet">
          <span>{tweet.text}</span>
        </div>
        <div className="buttons">
          <button onClick={() => alright(tweet)} disabled={tweet.idx < 0}>Alright</button>
          <button onClick={() => malicious(tweet)} disabled={tweet.idx < 0}>Malicious</button>
          <button onClick={() => console.log(`previous`)} disabled={tweet.idx < 0}>&lt;</button>
          <button onClick={() => console.log(`next`)} disabled={tweet.idx < 0}>&gt;</button>
          <button onClick={() => refresh()}>↻</button>
          <span>{uncertainty * 100}%</span>
        </div>
        <p>
          {progress} out of {total} data points labelled.
        </p>
        <p>
          Current classification performance (f1-score): {score * 100}% 
        </p>
      </div>
    </div>
  );
}

export default App;
