import { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Label, ReferenceLine, ResponsiveContainer } from 'recharts';
import { Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle, Button, ThemeProvider } from '@material-ui/core/';
import Loader from "react-loader-spinner";
import { Html5Entities } from 'html-entities';
import { socket } from './App';
import ReportTable from './ReportTable'

const htmlEntities = new Html5Entities();
const crypto = require('crypto');


function skip(tweet) {
  console.log(`"${tweet.text}" --> skipped`);
  socket.emit("skip", { 'idx': tweet.idx, 'hash': crypto.createHash('md5').update(tweet.text).digest('hex') });
}

function refresh(addToast) {
  console.log('refresh');
  socket.emit("refresh");
  addToast("Requested up-to-date information from the server.", { appearance: 'success' })
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

function checkpoint(addToast) {
  socket.emit("checkpoint", function (data) {
    if (data !== "Model is not initialised.") {
      addToast(data, { appearance: 'success' })
    } else {
      addToast(data, { appearance: 'error' })
    }
  }
  )
}

function Model({ model, setModel, loading, setLoading, theme, addToast }) {

  function label(tweet, label) {
    console.log(`"${tweet.text}" --> ${label}`);
    let sent = true
    setTimeout(() => { if (sent) setLoading(true) }, 200);
    socket.emit("label", { 'idx': tweet.idx, 'label': label, 'hash': crypto.createHash('md5').update(tweet.text).digest('hex') },
      (done) => {
        if (typeof done === "string") {
          addToast(done, { appearance: 'error' })
        }
        sent = false
        setLoading(false)
      });
  }

  const [resetDialog, setResetDialog] = useState(false);

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
          score: data.score * 100
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
          targetScore: data.target ? data.target : m.targetScore
        }
      })
    });
  }, [setModel]);

  return (
    <ThemeProvider theme={theme}>
      {model.tweet.idx < 0 ? null :
        <>
          <span><b>{model.progress}</b> out of <b>{model.total}</b> data points labelled</span>
          <span><b>{(model.uncertainty * 100).toFixed(2)}%</b> uncertainty</span>
        </>
      }
      {!model.targets.length || model.tweet.idx < 0 ? null :
        <div className="buttons">
          {Object.values(model.targets).map((target) =>
            <button style={{ minHeight: 40 }} className="button" key={target.val} onClick={() => label(model.tweet, target.val)}>{target.name}</button>
          )}
        </div>
      }
      <div className="tweet">
        <span>{htmlEntities.decode(model.tweet.text)}</span>
        <Loader type="TailSpin" color="#8884d8" height={25} width={25} visible={loading} />
      </div>
      <div className="buttons">
        <button className="button" onClick={() => skip(model.tweet)} disabled={model.tweet.idx < 0}>Skip</button>
        <button className="button" onClick={() => save(model.scoreSeries)} disabled={!model.scoreSeries.length}>Save</button>
        <button className="button" onClick={() => checkpoint(addToast)} disabled={model.tweet.idx < 0}>⚑</button>
        <button className="button" onClick={() => refresh(addToast)}>↻</button>
        <button className="button" onClick={() => setResetDialog(true)}>Reset</button>
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
            {!model.scoreSeries.length ? null :
              <ReportTable scores={model.scoreSeries} />}
          </div>
        </div>
      }
      <Dialog
        open={resetDialog}
        onClose={() => setResetDialog(false)}
        aria-labelledby="alert-dialog-title"
        aria-describedby="alert-dialog-description"
      >
        <DialogTitle id="alert-dialog-title">{"Are you sure?"}</DialogTitle>
        <DialogContent>
          <DialogContentText id="alert-dialog-description">
            This will reset the current active learning process and model.<br />
            All labels and statistics will be lost in the process unless previously saved.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => {
            socket.emit('reset', (reset) => {
              setModel(m => {
                return { ...m, initialised: (!reset) }
              })
              addToast("Classification model was successfully reset.", { appearance: 'success' })
            }
            );
          }} color="primary">
            Proceed
          </Button>
          <Button onClick={() => setResetDialog(false)} color="primary" autoFocus>
            Cancel
          </Button>
        </DialogActions>
      </Dialog>
    </ThemeProvider>
  );
}

export default Model;


