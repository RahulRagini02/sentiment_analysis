import React from 'react';
import './Header.css';

const Header = () => {
  return (
    <header className="header">
      <div className="logo">
        <span className="logo-icon">📊</span>
        SentimentAI
      </div>
      <nav>
        <a href="#home">Home</a>
        <a href="#demo">Demo</a>
        <a href="#upload">Upload File</a>
        <a href="#api">API Docs</a>
        <a href="#about">About</a>
      </nav>
    </header>
  );
};

export default Header;