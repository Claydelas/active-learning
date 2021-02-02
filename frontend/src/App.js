import { useState } from 'react';
import './App.css';

function App() {

  const [tweet, setTweet] = useState("Tweet text will be displayed here");
  const [uncertainty, setUncertainty] = useState(0.00);
  const [progress, setProgress] = useState(0);
  const [total, setTotal] = useState(0);
  const [score, setScore] = useState(0.00);

  return (
    <div className="App">
      <div className="App-main">
        <div className="tweet">
          <span>{tweet}</span>
        </div>
        <div className="buttons">
          <button onClick={() => setTweet("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aliquam ultricies sollicitudin lacus eu egestas. Aenean vel metus fermentum turpis.")}>demo</button>
          <button onClick={() => console.log(`"${tweet}" --> alright. --> 0`)}>Alright</button>
          <button onClick={() => console.log(`"${tweet}" --> malicious. --> 1`)}>Malicious</button>
          <button onClick={() => console.log(`previous`)}>&lt;</button>
          <button onClick={() => console.log(`next`)}>&gt;</button>
          <button onClick={() => console.log(`refresh`)}>↻</button>
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
