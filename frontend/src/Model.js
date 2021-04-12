import { useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Label, ReferenceLine, ResponsiveContainer } from 'recharts';
import { Html5Entities } from 'html-entities';
import { socket } from './App';

const htmlEntities = new Html5Entities();
const crypto = require('crypto');

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

function Model({ model, setModel }) {

  // on component mount listen for queries
  useEffect(() => {
    socket.on("query", data => {
      setModel(m => {
        return {
          ...m,
          tweet: { idx: data.idx, text: data.text },
          uncertainty: data.uncertainty,
          scoreSeries: JSON.stringify(data.series) !== '{}' ? data.series : m.scoreSeries,
          progress: data.labeled_size,
          total: data.dataset_size,
          score: data.score * 100,
          report: JSON.stringify(data.report) !== '{}' ? data.report : m.report
        }
      })
    });
  }, [setModel]);


  // on component mount listen for end
  useEffect(() => {
    socket.on("end", data => {
      setModel(m => {
        return {
          ...m,
          tweet: { idx: -1, text: "All samples labeled." },
          scoreSeries: JSON.stringify(data.series) !== '{}' ? data.series : m.scoreSeries,
          progress: data.labeled_size,
          total: data.dataset_size ? data.dataset_size : m.total,
          score: data.score * 100,
          targetScore: data.target ? data.target : m.targetScore,
          report: JSON.stringify(data.report) !== '{}' ? data.report : m.report
        }
      })
    });
  }, [setModel]);

  return (
    <>
      {model.tweet.idx < 0 ? null :
        <>
          <span><b>{model.progress}</b> out of <b>{model.total}</b> data points labelled</span>
          <span><b>{(model.uncertainty * 100).toFixed(2)}%</b> uncertainty</span>
        </>
      }
      {!model.targets.length || model.tweet.idx < 0 ? null :
        <div className="buttons">
          {Object.values(model.targets).map((target) =>
            <button className="button" key={target.val} onClick={() => label(model.tweet, target.val)}>{target.name}</button>
          )}
        </div>
      }
      <div className="tweet">
        <span>{htmlEntities.decode(model.tweet.text)}</span>
      </div>
      <div className="buttons">
        <button className="button" onClick={() => skip(model.tweet)} disabled={model.tweet.idx < 0}>Skip</button>
        <button className="button" onClick={() => save(model.scoreSeries)} disabled={!model.scoreSeries.length}>Save</button>
        <button className="button" onClick={() => checkpoint()} disabled={model.tweet.idx < 0}>⚑</button>
        <button className="button" onClick={() => refresh()}>↻</button>
      </div>
      {!model.scoreSeries.length ? null :
        <div className="stats">
          <ResponsiveContainer width="50%" height={500} className="graph">
            <LineChart
              width={500}
              height={300}
              data={model.scoreSeries}
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
              <ReferenceLine y={model.targetScore} stroke="#8884d8">
                <Label value="Learning Target" fill="#8884d8" position="top" />
              </ReferenceLine>
              <Line type="monotone" dataKey={data => { return data['macro avg']['f1-score'] * 100 }} stroke="#82ca9d" dot={false} activeDot={{ r: 8 }} />
            </LineChart>
          </ResponsiveContainer>
          <div>
            <p>
              Current classification performance: <b>{model.score.toFixed(2)}%</b>
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
                {Object.entries(model.report).map((key) =>
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
        </div>
      }
    </>
  );
}

export default Model;


