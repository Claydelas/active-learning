import { useState, useEffect } from 'react';
import './App.css';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Label, ReferenceLine, ResponsiveContainer } from 'recharts';
import { Html5Entities } from 'html-entities';

const htmlEntities = new Html5Entities();
const crypto = require('crypto');
const io = require("socket.io-client");
const socket = io('http://127.0.0.1:5000', { transports: ['websocket'] });

function label(tweet, label) {
  console.log(`"${tweet.text}" --> ${label}`);
  socket.emit("label", { 'idx': tweet.idx, 'label': label, 'hash': crypto.createHash('md5').update(tweet.text).digest('hex') });
}

function skip(tweet) {
  console.log(`"${tweet.text}" --> skipped`);
  socket.emit("skip", { 'idx': tweet.idx, 'hash': crypto.createHash('md5').update(tweet.text).digest('hex') });
}

function refresh() {
  console.log('refresh');
  socket.emit("refresh");
}

function save(scores) {
  console.log('saving performance graph');
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(scores, null, 2)], {
    type: "text/plain"
  }));
  a.setAttribute("download", `scores-${new Date().toISOString()}.json`);
  a.click();
}

function checkpoint() {
  socket.emit("checkpoint", function (data) {
    console.log(data);
  })
}

function App() {

  const [tweet, setTweet] = useState({ idx: -1, text: "Tweet text will be displayed here" }); // queried tweet object
  const [targets, setTargets] = useState([{ 'val': 0, 'name': 'non-Malicious' }, { 'val': 1, 'name': 'Malicious' }]); // label choices
  const [uncertainty, setUncertainty] = useState(0.00); // uncertainty of currently displayed tweet
  const [progress, setProgress] = useState(0); // number of labeled data points in pool
  const [total, setTotal] = useState(0); // total number of data points in pool
  const [score, setScore] = useState(0.00); // current/latest f1-score
  const [report, setReport] = useState({}); // classification report
  const [targetScore, setTargetScore] = useState(80.00); // threshold where active learning score aligns with non-AL
  const [scoreSeries, setScoreSeries] = useState([]); // model performance series

  // on component mount listen for queries
  useEffect(() => {
    socket.on("query", data => {
      console.log(data);
      setTweet({ idx: data.idx, text: data.text });
      setUncertainty(data.uncertainty);
      if (JSON.stringify(data.series) !== '{}') {
        setScoreSeries(score =>
          (score.length > 0 && score[score.length - 1]['labels'] !== data.series['labels']) || (!score.length)
            ? [...score, data.series]
            : [...score]);
      }
      setProgress(data.labeled_size);
      setTotal(data.dataset_size);
      setScore(data.score * 100);
      if (JSON.stringify(data.report) !== '{}') setReport(data.report);
    });
  }, []);

  // on component mount listen for init
  useEffect(() => {
    socket.on("init", data => {
      console.log(data);
      setTweet({ idx: data.idx, text: data.text });
      if (data.targets) setTargets(data.targets)
      setUncertainty(data.uncertainty);
      if (JSON.stringify(data.series) !== '{}') setScoreSeries(data.series);
      setProgress(data.labeled_size);
      setTotal(data.dataset_size);
      setScore(data.score * 100);
      setTargetScore(data.target);
      if (JSON.stringify(data.report) !== '{}') setReport(data.report);
    });
  }, []);

  // on component mount listen for end
  useEffect(() => {
    socket.on("end", data => {
      console.log(data);
      setTweet({ idx: -1, text: "All samples labeled." });
      if (JSON.stringify(data.series) !== '{}') setScoreSeries(data.series);
      setProgress(data.labeled_size);
      if (data.dataset_size) setTotal(data.dataset_size);
      setScore(data.score * 100);
      if (data.target) setTargetScore(data.target);
      if (JSON.stringify(data.report) !== '{}') setReport(data.report);
    });
  }, []);

  return (
    <div className="App">
      <div className="App-main">
        {tweet.idx < 0 ? null :
          <>
            <span><b>{progress}</b> out of <b>{total}</b> data points labelled</span>
            <span><b>{(uncertainty * 100).toFixed(2)}%</b> uncertainty</span>
          </>
        }
        {!targets.length || tweet.idx < 0 ? null :
          <div className="buttons">
            {Object.values(targets).map((target) =>
              <button className="button" key={target.val} onClick={() => label(tweet, target.val)}>{target.name}</button>
            )}
          </div>
        }
        <div className="tweet">
          <span>{htmlEntities.decode(tweet.text)}</span>
        </div>
        <div className="buttons">
          <button className="button" onClick={() => skip(tweet)} disabled={tweet.idx < 0}>Skip</button>
          <button className="button" onClick={() => save(scoreSeries)} disabled={!scoreSeries.length}>Save</button>
          <button className="button" onClick={() => checkpoint()} disabled={tweet.idx < 0}>⚑</button>
          <button className="button" onClick={() => refresh()}>↻</button>
        </div>
        <div className="stats">
          <ResponsiveContainer width="50%" height={500} className="graph">
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
                domain={['dataMin', 'dataMax']}
                allowDecimals={false}
                type="number"
                padding={{ left: 20, right: 20 }}>
                <Label value="number of labels" offset={-30} position="insideBottom" fill="#82ca9d" />
              </XAxis>
              <YAxis type="number"
                domain={[0, 100]}
                tickFormatter={tick => `${tick}%`}>
                <Label value="f1-score" angle={-90} offset={-30} position="insideLeft" fill="#82ca9d" />
              </YAxis>
              <Tooltip formatter={score => [`${score}%`, "f1-score"]}
                labelStyle={{ color: "#282c34" }}
                labelFormatter={label => `labels: ${label},`}
                contentStyle={{ borderRadius: "9px", fontSize: "18px", backgroundColor: "rgba(248, 248, 248, 0.85)", lineHeight: "20px" }}
                itemStyle={{ color: "#282c34" }} />
              <ReferenceLine y={targetScore} stroke="#8884d8">
                <Label value="Learning Target" fill="#8884d8" position="top" />
              </ReferenceLine>
              <Line type="monotone" dataKey={data => { return data['macro avg']['f1-score'] * 100 }} stroke="#82ca9d" dot={false} activeDot={{ r: 8 }} />
            </LineChart>
          </ResponsiveContainer>
          {!scoreSeries.length ? null :
            <div>
              <p>
                Current classification performance: <b>{score.toFixed(2)}%</b>
              </p>
              <table>
                <thead>
                  <tr>
                    <th></th>
                    <th>precision</th>
                    <th>recall</th>
                    <th>f1-score</th>
                    <th>support</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(report).map((key) =>
                    <tr key={key[0]}>
                      {(() => {
                        let i = 0;
                        switch (key[0]) {
                          case "accuracy": return (
                            [<td key={0}>{key[0]}</td>,
                            <td key={1}></td>,
                            <td key={2}></td>,
                            <td key={3}>{key[1].toFixed(2)}</td>,
                            <td key={4}></td>]);
                          case "labels": return null
                          default: return (
                            [<td key={key[0]}>{key[0]}</td>,
                            Object.entries(key[1]).map((val) => (
                              <td key={i++}>{val[0] === "support" ? val[1] : val[1].toFixed(2)}</td>
                            ))]);
                        }
                      })()}
                    </tr>)}
                </tbody>
              </table>
            </div>
          }
        </div>
      </div>
    </div>
  );
}

export default App;
