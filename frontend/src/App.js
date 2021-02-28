import { useState, useEffect } from 'react';
import './App.css';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Label, ReferenceLine, ResponsiveContainer } from 'recharts';

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
  const [targetScore, setTargetScore] = useState(80.00);
  const [scoreSeries, setScoreSeries] = useState([]);

  useEffect(() => {
    socket.on("tweet", data => {
      setTweet(data)
    });
  }, []);

  useEffect(() => {
    socket.on("score-data", data => {
      setScoreSeries(score => [...score, data]);
    });
  }, []);

  return (
    <div className="App">
      <div className="App-main">
        <p>
          {progress} out of {total} data points labelled
        </p>
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
          Current classification performance (f1-score): {score * 100}% 
        </p>
        <ResponsiveContainer width="50%" height="50%">
          <LineChart
            width={500}
            height={300}
            data={scoreSeries}
            margin={{
              top: 10,
              right: 45,
              left: 45,
              bottom: 45,
            }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="labels"
              type="number" 
              domain={['dataMin', 'dataMax']}
              tickFormatter={tick => `${tick}%`}
              padding={{ left: 20, right: 20 }}>
              <Label value="labeled ratio" offset={-30} position="insideBottom" fill="#82ca9d" />
            </XAxis>
            <YAxis type="number"
              domain={[0, 100]}
              tickFormatter={tick => `${tick}%`}>
              <Label value="f1-score" angle={-90} offset={-30} position="insideLeft" fill="#82ca9d" />
            </YAxis>
              <Tooltip formatter={score => [`${score}%`, "f1-score"]}
              labelStyle={{color: "#282c34"}}
              labelFormatter={label => `labeled: ${label}%,`}
              contentStyle={{borderRadius: "9px", fontSize: "18px", backgroundColor: "rgba(248, 248, 248, 0.85)", lineHeight: "20px"}}
              itemStyle={{color: "#282c34"}}/>
            <ReferenceLine y={targetScore} stroke="#8884d8">
              <Label value="Learning Target" fill="#8884d8" position="top" />
            </ReferenceLine>
            <Line type="monotone" dataKey="score" stroke="#82ca9d" activeDot={{ r: 8 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default App;
