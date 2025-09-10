import React, { useState } from "react";
import axios from "axios";

// Define your VERIFY API endpoint
const VERIFY_API = "http://127.0.0.1:8000/verify_key";

const KeyInput = ({ onKeyVerified }) => {
  const [apiKey, setApiKey] = useState("");
  const [status, setStatus] = useState("");

  const handleVerifyKey = async () => {
    try {
      const response = await axios.get(VERIFY_API, {
        headers: { "x-api-key": apiKey },
      });

      if (response.status === 200) {
        onKeyVerified(apiKey);
        setStatus("✅ API key verified successfully!");
      }
    } catch (error) {
      if (error.response && error.response.status === 401) {
        setStatus("❌ Invalid API key. Please try again.");
      } else {
        setStatus("⚠️ Unable to connect to API. Please check the backend server.");
      }
    }
  };

  return (
    <div className="card">
      <h3>API Key Verification</h3>
      <p>Please enter your API key to access the application features.</p>

      <input
        type="password"
        className="input-field"
        placeholder="Enter your API key here..."
        value={apiKey}
        onChange={(e) => setApiKey(e.target.value)}
      />

      <button className="button" onClick={handleVerifyKey}>
        Verify Key
      </button>

      {status && (
        <p style={{ color: status.startsWith("✅") ? "green" : "red" }}>
          {status}
        </p>
      )}
    </div>
  );
};

export default KeyInput;
