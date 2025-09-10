import React, { useState } from 'react';
import Header from './components/Header';
import Predictor from './components/Predictor';
import Uploader from './components/Uploader';
import Dashboard from './components/Dashboard';
import KeyInput from './components/KeyInput';
import './App.css';

function App() {
  const [keyVerified, setKeyVerified] = useState(false);
  const [api_key, setApiKey] = useState('');
  const [classifiedData, setClassifiedData] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);

  const handleKeyVerification = (key) => {
    setApiKey(key);
    setKeyVerified(true);
  };

  const handleClassifiedData = (data) => {
    setClassifiedData(data);
    setAnalysisData(null); // Clear analysis on new upload
  };

  const handleAnalysisData = (data) => {
    setAnalysisData(data);
  };

  return (
    <div className="app">
      <Header />
      <div className="content">
        <KeyInput onKeyVerified={handleKeyVerification} />
        {keyVerified && (
          <>
            <Predictor api_key={api_key} />
            <Uploader 
              api_key={api_key} 
              onClassified={handleClassifiedData}
              onAnalyzed={handleAnalysisData}
            />
            {classifiedData && (
              <Dashboard 
                api_key={api_key} 
                classifiedData={classifiedData}
                analysisData={analysisData}
                onAnalyzed={handleAnalysisData}
              />
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default App;