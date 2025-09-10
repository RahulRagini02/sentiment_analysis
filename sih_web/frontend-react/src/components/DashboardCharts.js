import React from 'react';

const DashboardCharts = ({ analysisData }) => {
  return (
    <div>
      <h3>Analysis & Visuals</h3>
      <div className="chart-container">
        {analysisData.bar_chart_base64 && (
          <div className="chart">
            <h4>Sentiment Distribution</h4>
            <img src={`data:image/png;base64,${analysisData.bar_chart_base64}`} alt="Sentiment Bar Chart" />
          </div>
        )}
        {analysisData.pie_chart_base64 && (
          <div className="chart">
            <h4>Sentiment Proportions</h4>
            <img src={`data:image/png;base64,${analysisData.pie_chart_base64}`} alt="Sentiment Pie Chart" />
          </div>
        )}
        {analysisData.wordcloud_base64 && (
          <div className="chart">
            <h4>Word Cloud</h4>
            <img src={`data:image/png;base64,${analysisData.wordcloud_base64}`} alt="Word Cloud" />
          </div>
        )}
        {analysisData.top_words_base64 && (
          <div className="chart">
            <h4>Top Words</h4>
            <img src={`data:image/png;base64,${analysisData.top_words_base64}`} alt="Top Words Chart" />
          </div>
        )}
        {analysisData.top_bigrams_base64 && (
          <div className="chart">
            <h4>Top Bigrams</h4>
            <img src={`data:image/png;base64,${analysisData.top_bigrams_base64}`} alt="Top Bigrams Chart" />
          </div>
        )}
        {analysisData.top_trigrams_base64 && (
          <div className="chart">
            <h4>Top Trigrams</h4>
            <img src={`data:image/png;base64,${analysisData.top_trigrams_base64}`} alt="Top Trigrams Chart" />
          </div>
        )}
        {analysisData.word_count_kde_base64 && (
          <div className="chart">
            <h4>Word Count Distribution</h4>
            <img src={`data:image/png;base64,${analysisData.word_count_kde_base64}`} alt="Word Count KDE Plot" />
          </div>
        )}
      </div>
    </div>
  );
};

export default DashboardCharts;