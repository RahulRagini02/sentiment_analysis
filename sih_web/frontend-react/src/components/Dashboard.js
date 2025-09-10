import React from 'react';
import { AgGridReact } from 'ag-grid-react';
import DashboardCharts from './DashboardCharts';
import axios from 'axios';

const ANALYZE_API = "http://127.0.0.1:8500/analyze";

const Dashboard = ({ api_key, classifiedData, analysisData, onAnalyzed }) => {

  const handleAnalyze = async () => {
    // The backend expects a file, so we convert the data to a CSV string
    const headers = Object.keys(classifiedData[0]).join(',');
    const csvContent = [headers, ...classifiedData.map(row => Object.values(row).join(','))].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const formData = new FormData();
    formData.append('file', blob, 'classified_data.csv');
    
    try {
      const response = await axios.post(ANALYZE_API, formData, {
        headers: { 'x-api-key': api_key },
      });
      onAnalyzed(response.data);
    } catch (error) {
      console.error("Analysis failed:", error);
      alert("Analysis failed. Please check the backend console for details.");
    }
  };

  const columnDefs = classifiedData && classifiedData.length > 0
    ? Object.keys(classifiedData[0]).map(key => ({ field: key }))
    : [];

  return (
    <div className="card">
      <h3>Preview of Labeled Data</h3>
      <div className="ag-theme-alpine" style={{ height: 400, width: '100%' }}>
        <AgGridReact
          rowData={classifiedData}
          columnDefs={columnDefs}
          defaultColDef={{ resizable: true }}
        />
      </div>
      <button className="button" onClick={handleAnalyze}>Run Analysis</button>

      {analysisData && <DashboardCharts analysisData={analysisData} />}
    </div>
  );
};

export default Dashboard;