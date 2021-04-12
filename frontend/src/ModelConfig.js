import { useState, useEffect } from 'react';
import { socket } from './App';

import { FormControl, MenuItem, InputLabel, Select, FormGroup, FormControlLabel, Checkbox, FormHelperText } from '@material-ui/core';
import { ThemeProvider, responsiveFontSizes, makeStyles } from '@material-ui/core/styles';
import { unstable_createMuiStrictModeTheme as createMuiTheme } from '@material-ui/core';
import Loader from "react-loader-spinner";

const withTimeout = (onSuccess, onTimeout, timeout) => {
  let called = false;

  const timer = setTimeout(() => {
    if (called) return;
    called = true;
    onTimeout();
  }, timeout);

  return (...args) => {
    if (called) return;
    called = true;
    clearTimeout(timer);
    onSuccess.apply(this, args);
  }
}

const darkTheme = responsiveFontSizes(createMuiTheme({
  palette: {
    type: 'dark',
    primary: { main: '#8884d8' },
    secondary: { main: '#82ca9d' },
    background: { paper: "#2c3039" },
    action: { selected: "#6c69ac" }
  },
}));

const useStyles = makeStyles((theme) => ({
  select: {
    minWidth: 400,
    fontWeight: 400,
    borderStyle: 'none',
    borderWidth: 2,
    borderRadius: 12,
    boxShadow: '0px 5px 8px -3px rgba(0,0,0,0.14)',
    "&:focus": {
      borderRadius: 12,
      background: '#282C34'
    }
  },
  paper: {
    borderRadius: 12,
    marginTop: 8
  },
  list: {
    paddingTop: 0,
    paddingBottom: 0,
    "& li": {
      fontWeight: 300,
      paddingTop: 12,
      paddingBottom: 12,
    },
    "& li:hover": {
      background: "#8884d8"
    },
  }
}));

function ModelConfig() {
  const classes = useStyles();
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
    socket.on("disconnect", () => {
      setClassifier("")
      setDataset("")
      setVectorizer("")
      setQueryStrategy("")
    })
  }, [])

  const [classifier, setClassifier] = useState("")
  const [dataset, setDataset] = useState("")
  const [vectorizer, setVectorizer] = useState("")
  const [queryStrategy, setQueryStrategy] = useState("")
  const [features, setFeatures] = useState({ text: false, user: false, stats: false })
  const [error, setError] = useState({ classifier: false, dataset: false, vectorizer: false, queryStrategy: false, features: false })
  const [loading, setLoading] = useState(false)

  const menuProps = {
    classes: {
      paper: classes.paper,
      list: classes.list
    },
    anchorOrigin: {
      vertical: "bottom",
      horizontal: "left"
    },
    transformOrigin: {
      vertical: "top",
      horizontal: "left"
    },
    getContentAnchorEl: null
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <Loader type="TailSpin" color="#8884d8" height={80} width={80} visible={loading} />
      <FormControl variant="outlined" error={error.classifier}>
        <InputLabel id="classifier">Classifier</InputLabel>
        <Select
          labelId="classifier"
          id="classifier_input"
          value={classifier}
          label="Classifier"
          MenuProps={menuProps}
          onChange={(e) => setClassifier(e.target.value)}
          classes={{ root: classes.select }}
        >
          {!options.classifiers.length ?
            <MenuItem value="">None</MenuItem> : options.classifiers.map((c) => <MenuItem key={c} value={c}>{c}</MenuItem>)}
        </Select>
      </FormControl>
      <FormControl variant="outlined" error={error.dataset}>
        <InputLabel id="dataset">Dataset</InputLabel>
        <Select
          labelId="dataset"
          id="dataset_input"
          value={dataset}
          label="Dataset"
          MenuProps={menuProps}
          onChange={(e) => setDataset(e.target.value)}
          classes={{ root: classes.select }}
        >
          {!options.datasets.length ?
            <MenuItem value="">None</MenuItem> : options.datasets.map((d) => <MenuItem key={d} value={d}>{d}</MenuItem>)}
        </Select>
      </FormControl>
      <FormControl variant="outlined" error={error.vectorizer}>
        <InputLabel id="vectorizer">Vectorizer</InputLabel>
        <Select
          labelId="vectorizer"
          id="vectorizer_input"
          value={vectorizer}
          label="Vectorizer"
          MenuProps={menuProps}
          onChange={(e) => setVectorizer(e.target.value)}
          classes={{ root: classes.select }}
        >
          {!options.vectorizers.length ?
            <MenuItem value="">None</MenuItem> : options.vectorizers.map((v) => <MenuItem key={v} value={v}>{v}</MenuItem>)}
        </Select>
      </FormControl>
      <FormControl variant="outlined" error={error.queryStrategy}>
        <InputLabel id="query_strategy">Query Strategy</InputLabel>
        <Select
          labelId="query_strategy"
          id="query_strategy_input"
          value={queryStrategy}
          label="Query Strategy"
          MenuProps={menuProps}
          onChange={(e) => setQueryStrategy(e.target.value)}
          classes={{ root: classes.select }}
        >
          {!options.query_strategies.length ?
            <MenuItem value="">None</MenuItem> : options.query_strategies.map((q) => <MenuItem key={q} value={q}>{q}</MenuItem>)}
        </Select>
      </FormControl>
      <FormControl required error={error.features}>
        <FormGroup row={true}>
          <FormControlLabel
            control={<Checkbox checked={features.text}
              onChange={(e) => setFeatures({ ...features, [e.target.name]: e.target.checked })}
              color="primary" name="text" />}
            label="Text Features"
          />
          <FormControlLabel
            control={<Checkbox checked={features.user}
              onChange={(e) => setFeatures({ ...features, [e.target.name]: e.target.checked })}
              color="primary" name="user" />}
            label="User Features"
          />
          <FormControlLabel
            control={<Checkbox checked={features.stats}
              onChange={(e) => setFeatures({ ...features, [e.target.name]: e.target.checked })}
              color="primary" name="stats" />}
            label="Text Statistics"
          />
        </FormGroup>
        <FormHelperText style={{ marginLeft: 'auto' }}>select at least 1 category</FormHelperText>
      </FormControl>
      <button className="button"
        disabled={loading}
        onClick={() => {
          let errors = {
            classifier: !options.classifiers.includes(classifier),
            dataset: !options.datasets.includes(dataset),
            vectorizer: !options.vectorizers.includes(vectorizer),
            queryStrategy: !options.query_strategies.includes(queryStrategy),
            features: [features.text, features.user, features.stats].filter((v) => v).length < 1
          }
          if ([errors.classifier, errors.dataset, errors.vectorizer, errors.queryStrategy, errors.features].filter((v) => v).length > 0) {
            setError(errors)
          } else {
            setLoading(true)
            socket.emit("options", {
              classifier: classifier,
              dataset: dataset,
              vectorizer: vectorizer,
              query_strategy: queryStrategy,
              features: features
            }, withTimeout((success) => {
              setLoading(!success)
            }, () => { setLoading(false) }, 30000))
          }
        }}
      >Begin Learning</button>
    </ThemeProvider>
  );
}

export default ModelConfig;
