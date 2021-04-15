import './App.css';
import Model from './Model';
import ModelConfig from './ModelConfig';
import { useState, useEffect } from 'react';
import { responsiveFontSizes, unstable_createMuiStrictModeTheme as createMuiTheme } from '@material-ui/core/styles';


const io = require("socket.io-client");
export const socket = io('http://127.0.0.1:5000', { transports: ['websocket'] });

function App() {

    const [loading, setLoading] = useState(false)

    const [model, setModel] = useState({
        tweet: { idx: -1, text: "Tweet text will be displayed here" },
        targets: [{ 'val': 0, 'name': 'non-Malicious' }, { 'val': 1, 'name': 'Malicious' }],
        uncertainty: 0.00,
        progress: 0,
        total: 0,
        score: 0.00,
        targetScore: 80.00,
        scoreSeries: [],
        initialised: false
    })

    const [options, setOptions] = useState({
        classifiers: [],
        datasets: [],
        vectorizers: [],
        query_strategies: []
    });

    useEffect(() => {
        socket.on("options", data => {
            setOptions(data)
        });
    }, []);

    useEffect(() => {
        socket.on("init", data => {
            setModel(m => {
                return {
                    ...m,
                    tweet: { idx: data.idx, text: data.text },
                    targets: data.targets ? data.targets : m.targets,
                    uncertainty: data.uncertainty,
                    scoreSeries: JSON.stringify(data.series) !== '{}' ? data.series : m.scoreSeries,
                    progress: data.labeled_size,
                    total: data.dataset_size,
                    score: data.score * 100,
                    targetScore: data.target,
                    initialised: true
                }
            })
        });
    }, []);

    useEffect(() => {
        socket.on("disconnect", () => {
            setModel(m => {
                return { ...m, initialised: false }
            })
            setOptions({
                classifiers: [],
                datasets: [],
                vectorizers: [],
                query_strategies: []
            })
        })
    }, [])

    const darkTheme = responsiveFontSizes(createMuiTheme({
        palette: {
            type: 'dark',
            primary: { main: '#8884d8' },
            secondary: { main: '#82ca9d' },
            background: { paper: "#2c3039" },
            action: { selected: "#6c69ac" }
        },
    }));

    return (
        <div className="App">
            <div className="App-main">
                {model.initialised ?
                    <Model
                        setModel={setModel}
                        model={model}
                        loading={loading}
                        setLoading={setLoading}
                        theme={darkTheme}
                    /> :
                    <ModelConfig
                        options={options}
                        loading={loading}
                        setLoading={setLoading}
                        theme={darkTheme}
                    />}
            </div>
        </div>
    )
}

export default App;