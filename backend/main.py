from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from typing import Dict, Any
import io
import uvicorn

app = FastAPI(title="Data Analytics API")

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
    Dummy endpoint for testing data analysis functionality.
    Returns fixed sample responses instead of actual processing.
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


@app.get("/")
async def root():
    return {
        "message": "Data Analytics API is running. Use /analyze endpoint to upload and analyze data."
    }


if __name__ == "__main__":

    # Run the server with CORS enabled
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
