# DataHammer Backend

This is the backend for the DataHammer application, which provides data analysis capabilities using a swarm of LlamaIndex agents powered by Google's Gemini Flash-Lite model.

## Architecture

The backend is built using a swarm of specialized agents, each responsible for a different part of the data pipeline:

1. **Supervisor Agent**: Coordinates the work of specialized agents and determines the flow of execution.
2. **Data Loading Agent**: Handles file uploads and initial data parsing.
3. **Data Cleaning Agent**: Handles data cleaning and preprocessing.
4. **Data Analysis Agent**: Performs statistical analysis and generates insights.
5. **Data Visualization Agent**: Creates visualizations based on the data.
6. **Reporting Agent**: Compiles insights and visualizations into a coherent report.

These agents communicate through a shared environment and use Google's Gemini Flash-Lite model for natural language understanding and generation.

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your Google AI Studio API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   MODEL_NAME=gemini-2.0-flash-lite
   HOST=0.0.0.0
   PORT=8000
   ```

3. Run the server:
   ```bash
   ./run.sh
   ```

## API Endpoints

- `GET /`: Root endpoint that returns a welcome message.
- `POST /analyze`: Upload and analyze a data file.
- `GET /status`: Get the current status of the pipeline.
- `GET /results`: Get the current results of the analysis.

## Example Usage

```python
import requests

# Upload a file for analysis
with open('data.csv', 'rb') as f:
    response = requests.post('http://localhost:8000/analyze', files={'file': f})
    
print(response.json())

# Check the status of the pipeline
status = requests.get('http://localhost:8000/status')
print(status.json())

# Get the results
results = requests.get('http://localhost:8000/results')
print(results.json())
```

## How It Works

1. When a file is uploaded to the `/analyze` endpoint, the Data Loading Agent loads the file and extracts basic information.
2. The Supervisor Agent creates a plan for processing the file through the pipeline.
3. The pipeline is executed in the background, with each agent performing its specialized task.
4. The results are continuously updated and can be accessed through the `/results` endpoint.
5. The status of the pipeline can be monitored through the `/status` endpoint.

The pipeline is non-linear, meaning that if issues are detected (e.g., visualization fails due to unclean data), the Supervisor Agent can go back to previous agents to address the issues.

## Requirements

- Python 3.9+
- FastAPI
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Google Generative AI Python SDK
- LlamaIndex
- Pydantic
- Python-dotenv