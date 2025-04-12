import './App.css';
import { useState } from 'react';

function App() {
  const [file, setFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileUpload = (event) => {
    const uploadedFile = event.target.files[0];
    if (uploadedFile) {
      setFile(uploadedFile);
    }
  };

  const handleAnalyze = () => {
    if (!file) {
      alert('Please upload a spreadsheet first');
      return;
    }
    
    setIsLoading(true);
    // Simulate analysis process
    setTimeout(() => {
      setIsLoading(false);
      alert('Analysis complete! Your insights report is ready.');
    }, 2000);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>DataHammer</h1>
        <p className="tagline">
          Powerful Data Analytics Made Simple
        </p>
        
        <div className="upload-container">
          <h2>Upload Your Data</h2>
          <p>Upload your spreadsheet and get instant insights</p>
          
          <div className="file-input-container">
            <input 
              type="file" 
              id="file-upload" 
              onChange={handleFileUpload}
              accept=".csv,.xlsx,.xls"
            />
            <label htmlFor="file-upload" className="file-input-label">
              {file ? file.name : 'Choose File'}
            </label>
          </div>
          
          <button 
            className="analyze-button" 
            onClick={handleAnalyze}
            disabled={isLoading || !file}
          >
            {isLoading ? 'Analyzing...' : 'Run Analysis'}
          </button>
        </div>
        
        <div className="features">
          <h3>Features</h3>
          <ul>
            <li>Drag and drop interface for data exploration</li>
            <li>Automated insights and pattern detection</li>
            <li>Interactive visualizations</li>
            <li>Export reports in multiple formats</li>
          </ul>
        </div>
      </header>
    </div>
  );
}

export default App;
