import React, { useState } from 'react';
import axios from 'axios';

const CLASSIFY_API = "http://127.0.0.1:8000/classify_file";

const Uploader = ({ api_key, onClassified }) => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file to upload.");
      return;
    }
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(CLASSIFY_API, formData, {
        headers: { 'x-api-key': api_key, 'Content-Type': 'multipart/form-data' },
        responseType: 'blob'
      });
      
      const classifiedCsv = new Blob([response.data], { type: 'text/csv' });
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target.result;
        const df = parseCsv(text);
        onClassified(df);
      };
      reader.readAsText(classifiedCsv);

    } catch (error) {
      console.error("File upload failed:", error);
      alert("File upload failed. Check the API key and file format.");
    }
    setLoading(false);
  };

  const parseCsv = (csvText) => {
    const lines = csvText.split('\n');
    const headers = lines[0].split(',').map(header => header.trim());
    const data = lines.slice(1).map(line => {
      const values = line.split(',').map(val => val.trim());
      const row = {};
      headers.forEach((header, i) => {
        row[header] = values[i];
      });
      return row;
    }).filter(row => Object.values(row).some(val => val !== ''));
    return data;
  };

  return (
    <div className="card">
      <h3>Upload File for Classification</h3>
      <input type="file" onChange={handleFileChange} />
      <button className="button" onClick={handleUpload} disabled={loading}>
        {loading ? 'Uploading...' : 'Upload & Classify'}
      </button>
    </div>
  );
};

export default Uploader;