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


@app.post("/agent-swarm/report")
async def generate_report(
    file: UploadFile = File(...),
    goals: List[str] = Body(["Generate comprehensive data analysis report"]),
    report_format: str = Body("jupyter")
):
    """
    Generate a report using the ReportingAgent.
    
    Args:
        file: The data file to analyze
        goals: List of goals for the report
        report_format: Format for the report (jupyter, markdown, html, pdf)
        
    Returns:
        Dict containing the report data
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
        
        # Create output directory for reports
        output_dir = os.path.join(tempfile.gettempdir(), "datahammer_reports")
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize the ReportingAgent
        reporting_agent = ReportingAgent(output_dir=output_dir)
        
        # Prepare mock environment data (in a real implementation, this would come from other agents)
        environment = {
            "Data Overview": {
                "summary": {
                    "datasets": [file.filename],
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
                }
            },
            "Cleaned Data": {
                "cleaning_steps": {
                    file.filename: [
                        {"operation": "Remove duplicates", "description": "Removed 3 duplicate rows"},
                        {"operation": "Fill missing values", "description": "Filled missing values in 'age' column with median"}
                    ]
                }
            },
            "Analysis Results": {
                "insights": {
                    file.filename: [
                        {"type": "Correlation", "description": "Strong correlation between age and salary", "details": "r=0.78"},
                        {"type": "Distribution", "description": "Salary is right-skewed", "details": "skewness=1.45"},
                        {"type": "Outliers", "description": "Found 2 outliers in salary column", "details": "Z-score > 3"}
                    ]
                },
                "findings": {
                    file.filename: {
                        "recommendations": [
                            "Consider log-transforming the salary column for modeling",
                            "Investigate the outliers in the salary column",
                            "Segment analysis by department for more insights"
                        ]
                    }
                }
            },
            "Models": {
                "trained_model": {
                    file.filename: {
                        "model_type": "regression",
                        "target": "salary",
                        "features": ["age", "department"]
                    }
                },
                "performance": {
                    file.filename: {
                        "best_model": "RandomForestRegressor",
                        "RandomForestRegressor": {
                            "metrics": {
                                "r2": 0.85,
                                "mae": 2500,
                                "rmse": 3200
                            }
                        }
                    }
                }
            },
            "Visualizations": {
                "plots": {
                    file.filename: [
                        "correlation_heatmap.png",
                        "salary_distribution.png",
                        "age_vs_salary.png"
                    ]
                },
                "dashboard": {
                    "sections": [
                        {"title": "Overview", "plots": ["data_summary.png"]},
                        {"title": "Correlations", "plots": ["correlation_heatmap.png"]},
                        {"title": "Distributions", "plots": ["salary_distribution.png", "age_distribution.png"]}
                    ]
                }
            }
        }
        
        # Run the ReportingAgent
        result = reporting_agent.run(
            environment=environment,
            goals=goals,
            report_format=report_format,
            output_dir=output_dir
        )
        
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        
        # Return the report data
        return {
            "status": "success",
            "report": {
                "summary": result.get("report_summary", {}),
                "content": result.get("report_content", {}),
                "recommendations": result.get("report_content", {}).get("recommendations", []),
                "visualizations": result.get("visualization_requests", []),
                "report_path": result.get("report", "")
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@app.get("/")
async def root():
    return {
        "message": "DataHammer Analytics API is running",
        "endpoints": {
            "/analyze": "Traditional data analysis",
            "/agent-swarm/analyze": "AI Agent Swarm data analysis",
            "/agent-swarm/agents": "List available agents",
            "/agent-swarm/report": "Generate report using the ReportingAgent"
        }
    }


if __name__ == "__main__":
    # Run the server with CORS enabled
    uvicorn.run("main:app", host="0.0.0.0", port=12001, reload=True)
