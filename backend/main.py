"""
Main API for the DataHammer backend.
"""

import os
import io
import json
import asyncio
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import uvicorn

from shared.state import SharedState, shared_state
from agents.supervisor_agent import SupervisorAgent
from agents.data_loading_agent import DataLoadingAgent
from agents.data_cleaning_agent import DataCleaningAgent
from agents.data_analysis_agent import DataAnalysisAgent
from agents.data_visualization_agent import DataVisualizationAgent
from agents.reporting_agent import ReportingAgent

# Load environment variables
load_dotenv()

# Create the FastAPI app
app = FastAPI(title="Data Analytics API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get API key from environment
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Warning: GOOGLE_API_KEY not found in environment variables.")

# Get model name from environment
model_name = os.getenv("MODEL_NAME", "gemini-2.0-flash-lite")

# Create the agents
supervisor = SupervisorAgent(shared_state, api_key=api_key, model_name=model_name)
data_loading_agent = DataLoadingAgent(shared_state, api_key=api_key, model_name=model_name)
data_cleaning_agent = DataCleaningAgent(shared_state, api_key=api_key, model_name=model_name)
data_analysis_agent = DataAnalysisAgent(shared_state, api_key=api_key, model_name=model_name)
data_visualization_agent = DataVisualizationAgent(shared_state, api_key=api_key, model_name=model_name)
reporting_agent = ReportingAgent(shared_state, api_key=api_key, model_name=model_name)

# Register the agents with the supervisor
supervisor.register_agent(data_loading_agent)
supervisor.register_agent(data_cleaning_agent)
supervisor.register_agent(data_analysis_agent)
supervisor.register_agent(data_visualization_agent)
supervisor.register_agent(reporting_agent)

# Background task for processing the pipeline
async def process_pipeline():
    """Process the data pipeline in the background."""
    try:
        await supervisor.execute_pipeline()
    except Exception as e:
        shared_state.add_error(f"Pipeline execution failed: {str(e)}")


@app.post("/analyze")
async def analyze_data(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Analyze a data file.
    
    Args:
        background_tasks: FastAPI background tasks.
        file: The file to analyze.
        
    Returns:
        The analysis results.
    """
    try:
        # Reset the shared state
        global shared_state
        shared_state = SharedState()
        
        # Read the file content
        file_content = await file.read()
        
        # Get file info
        file_info = {
            "filename": file.filename,
            "file_extension": file.filename.split('.')[-1].lower(),
            "file_size": len(file_content),
            "content_type": file.content_type,
        }
        
        # Load the file based on its extension
        if file_info["file_extension"] == "csv":
            result = data_loading_agent.load_csv(file_content, file.filename)
        elif file_info["file_extension"] in ["xls", "xlsx"]:
            result = data_loading_agent.load_excel(file_content, file.filename)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_info['file_extension']}")
        
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("message", "Failed to load file"))
        
        # Plan the pipeline
        plan_result = await supervisor.plan_pipeline(file_info)
        
        # Start the pipeline in the background
        background_tasks.add_task(process_pipeline)
        
        # Return the initial results
        return {
            "summary": {
                "rows": shared_state.dataset_info.rows,
                "columns": shared_state.dataset_info.columns,
                "column_names": shared_state.dataset_info.column_names,
                "dtypes": shared_state.dataset_info.dtypes,
                "missing_values": shared_state.dataset_info.missing_values,
            },
            "insights": [insight.text for insight in shared_state.insights],
            "visualizations": {
                viz.title.lower().replace(" ", "_"): viz.data
                for viz in shared_state.visualizations
            },
            "predictive_modeling": {
                "model_type": shared_state.predictive_models[0].model_type if shared_state.predictive_models else None,
                "target_column": shared_state.predictive_models[0].target_column if shared_state.predictive_models else None,
                "feature_importance": shared_state.predictive_models[0].feature_importance if shared_state.predictive_models else {},
                "top_features": shared_state.predictive_models[0].top_features if shared_state.predictive_models else [],
            },
            "pipeline_status": {
                "state": supervisor.pipeline_state,
                "current_task_id": supervisor.current_task_id,
                "pending_tasks": len(shared_state.get_pending_tasks()),
                "completed_tasks": len(shared_state.task_history),
                "errors": shared_state.errors,
                "warnings": shared_state.warnings,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/status")
async def get_status():
    """
    Get the current status of the pipeline.
    
    Returns:
        The current status.
    """
    return {
        "pipeline_status": {
            "state": supervisor.pipeline_state,
            "current_task_id": supervisor.current_task_id,
            "pending_tasks": len(shared_state.get_pending_tasks()),
            "completed_tasks": len(shared_state.task_history),
            "errors": shared_state.errors,
            "warnings": shared_state.warnings,
        },
        "dataset_info": {
            "rows": shared_state.dataset_info.rows,
            "columns": shared_state.dataset_info.columns,
            "column_names": shared_state.dataset_info.column_names,
            "dtypes": shared_state.dataset_info.dtypes,
            "missing_values": shared_state.dataset_info.missing_values,
        } if shared_state.dataframe is not None else None,
        "insights_count": len(shared_state.insights),
        "visualizations_count": len(shared_state.visualizations),
        "predictive_models_count": len(shared_state.predictive_models),
    }


@app.get("/results")
async def get_results():
    """
    Get the current results of the analysis.
    
    Returns:
        The current results.
    """
    return {
        "summary": {
            "rows": shared_state.dataset_info.rows,
            "columns": shared_state.dataset_info.columns,
            "column_names": shared_state.dataset_info.column_names,
            "dtypes": shared_state.dataset_info.dtypes,
            "missing_values": shared_state.dataset_info.missing_values,
        } if shared_state.dataframe is not None else None,
        "insights": [insight.model_dump() for insight in shared_state.insights],
        "visualizations": {
            viz.title.lower().replace(" ", "_"): viz.data
            for viz in shared_state.visualizations
        },
        "predictive_modeling": {
            "model_type": shared_state.predictive_models[0].model_type if shared_state.predictive_models else None,
            "target_column": shared_state.predictive_models[0].target_column if shared_state.predictive_models else None,
            "feature_importance": shared_state.predictive_models[0].feature_importance if shared_state.predictive_models else {},
            "top_features": shared_state.predictive_models[0].top_features if shared_state.predictive_models else [],
        } if shared_state.predictive_models else None,
    }


@app.get("/")
async def root():
    """
    Root endpoint.
    
    Returns:
        A welcome message.
    """
    return {
        "message": "Data Analytics API is running. Use /analyze endpoint to upload and analyze data."
    }


if __name__ == "__main__":
    # Get host and port from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    # Run the server
    uvicorn.run("main:app", host=host, port=port, reload=True)
