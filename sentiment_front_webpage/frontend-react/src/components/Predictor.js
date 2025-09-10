import React, { useState } from 'react';
import axios from 'axios';

const PREDICT_API = "http://127.0.0.1:8000/predict";

const Predictor = ({ api_key }) => {
  const [inputText, setInputText] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const response = await axios.post(PREDICT_API, { text: inputText }, {
        headers: { 'x-api-key': api_key }
      });
      setPrediction(response.data);
    } catch (error) {
      console.error("Prediction failed:", error);
      setPrediction({ error: "Prediction failed." });
    }
    setLoading(false);
  };

  const getSentimentColor = (sentiment) => {
    switch (sentiment.toLowerCase()) {
      case 'positive': return 'green';
      case 'negative': return 'red';
      case 'neutral': return 'orange';
      default: return 'gray';
    }
  };

  return (
    <div className="card">
      <h3>Analyze Emotions in Text</h3>
      <div className="input-group">
        <textarea 
          className="input-field"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Type or paste text here..."
        />
        <button className="button" onClick={handlePredict} disabled={loading}>
          {loading ? 'Predicting...' : 'Predict'}
        </button>
      </div>
      {prediction && (
        <div className="result-card">
          <h4>Prediction Result</h4>
          <p>Sentiment: <strong style={{ color: getSentimentColor(prediction.prediction) }}>{prediction.prediction}</strong></p>
          <p>Confidence: <strong>{Math.round(prediction.Confidence * 100)}%</strong></p>
        </div>
      )}
    </div>
  );
};

export default Predictor;