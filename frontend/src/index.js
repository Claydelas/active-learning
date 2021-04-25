import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import App from './App';
import { ToastProvider } from 'react-toast-notifications'

ReactDOM.render(
  <React.StrictMode>
    <ToastProvider autoDismissTimeout={4000} autoDismiss={true}>
      <App />
    </ToastProvider>
  </React.StrictMode>,
  document.getElementById('root')
);

