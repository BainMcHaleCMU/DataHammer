from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import io
import os
import tempfile
import uvicorn
import json

from agent_swarm.agents import (
    OrchestratorAgent,
    DataLoadingAgent,
    ExplorationAgent,
    CleaningAgent,
    AnalysisAgent,
    ModelingAgent,
    VisualizationAgent,
    CodeActAgent,
    ReportingAgent
)
from agent_swarm.main import setup_llm, create_agent_swarm, run_agent_swarm

app = FastAPI(title="DataHammer Analytics API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze")
async def analyze_data(file: UploadFile = File(...)):
    """
    Analyze data using the traditional approach.
    Returns fixed sample responses for now.
    """
    try:
        # Create dummy results
        results = {
            "summary": {
                "rows": 100,
                "columns": 5,
                "column_names": ["id", "name", "age", "salary", "department"],
                "dtypes": {
                    "id": "int64",
                    "name": "object",
                    "age": "int64",
                    "salary": "float64",
                    "department": "object",
                },
                "missing_values": {
                    "id": 0,
                    "name": 2,
                    "age": 5,
                    "salary": 3,
                    "department": 1,
                },
            },
            "insights": [
                "Found 11 missing values across 4 columns.",
                "Column 'salary' is positively skewed (1.45).",
                "Column 'name' has all unique values, might be an ID column.",
            ],
            "visualizations": {
                "correlation_heatmap": "base64_encoded_image_placeholder",
                "distribution_age": "base64_encoded_image_placeholder",
                "categorical_department": "base64_encoded_image_placeholder",
            },
            "predictive_modeling": {
                "model_type": "regression",
                "target_column": "salary",
                "feature_importance": {"age": 0.65, "id": 0.30, "department": 0.05},
                "top_features": ["age", "id", "department"],
            },
        }

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/agent-swarm/analyze")
async def agent_swarm_analyze(
    file: UploadFile = File(...),
    goals: List[str] = Body(["Explore the data", "Identify key insights", "Build a predictive model"])
):
    """
    Analyze data using the AI Agent Swarm.
    
    Args:
        file: The data file to analyze
        goals: List of goals for the analysis
        
    Returns:
        Dict containing the analysis results
    """
    try:
        # Save the uploaded file to a temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        temp_file_path = temp_file.name
        
        try:
            contents = await file.read()
            with open(temp_file_path, 'wb') as f:
                f.write(contents)
        except Exception as e:
            os.unlink(temp_file_path)
            raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")
        
        # Set up data sources
        data_sources = {
            "file_path": temp_file_path,
            "file_type": file.filename.split('.')[-1].lower()
        }
        
        # Run the agent swarm
        try:
            # This is a placeholder - in a real implementation, we would
            # actually run the agent swarm and return real results
            
            # Simulate agent swarm execution
            results = {
                "status": "success",
                "message": "Agent Swarm analysis completed",
                "goals": goals,
                "data_sources": data_sources,
                "notebook_path": "path/to/generated_notebook.ipynb",
                "summary": {
                    "insights": [
                        "The data contains 5 columns and 100 rows",
                        "There are missing values in the 'age' column",
                        "The 'salary' column shows a strong correlation with 'age'"
                    ],
                    "models": [
                        {
                            "name": "Random Forest Regressor",
                            "target": "salary",
                            "accuracy": 0.85,
                            "important_features": ["age", "department"]
                        }
                    ],
                    "visualizations": [
                        "path/to/correlation_matrix.png",
                        "path/to/feature_importance.png"
                    ]
                }
            }
            
            return results
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Agent Swarm analysis failed: {str(e)}")
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/agent-swarm/agents")
async def list_agents():
    """
    List all available agents in the AI Agent Swarm.
    
    Returns:
        Dict containing the list of available agents
    """
    agents = [
        {
            "name": "OrchestratorAgent",
            "description": "Central coordinator for the AI Agent Swarm"
        },
        {
            "name": "DataLoadingAgent",
            "description": "Agent responsible for ingesting data from various sources"
        },
        {
            "name": "ExplorationAgent",
            "description": "Agent responsible for performing initial data analysis and profiling"
        },
        {
            "name": "CleaningAgent",
            "description": "Agent responsible for preprocessing and cleaning data"
        },
        {
            "name": "AnalysisAgent",
            "description": "Agent responsible for deriving deeper insights from data"
        },
        {
            "name": "ModelingAgent",
            "description": "Agent responsible for developing and evaluating predictive models"
        },
        {
            "name": "VisualizationAgent",
            "description": "Agent responsible for generating visual representations"
        },
        {
            "name": "CodeActAgent",
            "description": "Agent responsible for securely executing Python code snippets"
        },
        {
            "name": "ReportingAgent",
            "description": "Agent responsible for compiling final results into a report"
        }
    ]
    
    return {"agents": agents}


@app.get("/")
async def root():
    return {
        "message": "DataHammer Analytics API is running",
        "endpoints": {
            "/analyze": "Traditional data analysis",
            "/agent-swarm/analyze": "AI Agent Swarm data analysis",
            "/agent-swarm/agents": "List available agents"
        }
    }


if __name__ == "__main__":
    # Run the server with CORS enabled
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
