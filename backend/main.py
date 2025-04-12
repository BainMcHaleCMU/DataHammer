from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import io
import json
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from typing import Dict, List, Any, Optional
import os

app = FastAPI(title="Data Analytics API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

def read_file(file: UploadFile) -> pd.DataFrame:
    """Read uploaded file into pandas DataFrame."""
    content = file.file.read()
    
    # Determine file type and read accordingly
    if file.filename.endswith('.csv'):
        return pd.read_csv(io.BytesIO(content))
    elif file.filename.endswith(('.xls', '.xlsx')):
        return pd.read_excel(io.BytesIO(content))
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or Excel file.")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic data cleaning."""
    # Drop rows with all NaN values
    df = df.dropna(how='all')
    
    # Fill remaining NaN values with column means for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())
    
    # Fill NaN values with mode for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    
    return df

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate summary statistics for the dataset."""
    # Basic info
    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
    }
    
    # Summary statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary["numeric_stats"] = json.loads(df[numeric_cols].describe().to_json())
    
    # Categorical column summaries
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        summary["categorical_stats"] = {
            col: df[col].value_counts().to_dict() for col in categorical_cols
        }
    
    return summary

def generate_visualizations(df: pd.DataFrame) -> Dict[str, str]:
    """Generate basic visualizations for the dataset."""
    visualizations = {}
    
    # Only process if we have numeric data
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation = df[numeric_cols].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        visualizations["correlation_heatmap"] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Distribution plots for numeric columns (limit to first 5)
        for i, col in enumerate(numeric_cols[:5]):
            plt.figure(figsize=(8, 6))
            sns.histplot(df[col], kde=True)
            plt.title(f'Distribution of {col}')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            visualizations[f"distribution_{col}"] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
    
    # Categorical data visualization (limit to first 5)
    categorical_cols = df.select_dtypes(include=['object']).columns
    for i, col in enumerate(categorical_cols[:5]):
        plt.figure(figsize=(10, 6))
        value_counts = df[col].value_counts().head(10)  # Top 10 categories
        sns.barplot(x=value_counts.index, y=value_counts.values)
        plt.title(f'Top 10 Categories in {col}')
        plt.xticks(rotation=45, ha='right')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        visualizations[f"categorical_{col}"] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
    
    return visualizations

def generate_insights(df: pd.DataFrame) -> List[str]:
    """Generate basic insights from the data."""
    insights = []
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        cols_with_missing = missing[missing > 0]
        insights.append(f"Found {missing.sum()} missing values across {len(cols_with_missing)} columns.")
    
    # Check for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        # Find highly correlated features
        corr_matrix = df[numeric_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = [(col1, col2, corr_matrix.loc[col1, col2]) 
                     for col1 in upper.index 
                     for col2 in upper.columns 
                     if upper.loc[col1, col2] > 0.8]
        
        if high_corr:
            insights.append(f"Found {len(high_corr)} pairs of highly correlated features (>0.8).")
            for col1, col2, corr in high_corr[:3]:  # Show top 3
                insights.append(f"  - {col1} and {col2} have correlation of {corr:.2f}")
        
        # Identify skewed distributions
        for col in numeric_cols:
            skew = df[col].skew()
            if abs(skew) > 1:
                insights.append(f"Column '{col}' is {'positively' if skew > 0 else 'negatively'} skewed ({skew:.2f}).")
    
    # Check for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            unique_count = df[col].nunique()
            total_count = len(df)
            if unique_count == total_count:
                insights.append(f"Column '{col}' has all unique values, might be an ID column.")
            elif unique_count == 1:
                insights.append(f"Column '{col}' has only one value: '{df[col].iloc[0]}'.")
            elif unique_count / total_count < 0.01:
                insights.append(f"Column '{col}' has high cardinality with {unique_count} unique values.")
    
    return insights

def run_predictive_modeling(df: pd.DataFrame) -> Dict[str, Any]:
    """Run basic predictive modeling on the dataset."""
    results = {}
    
    # Only proceed if we have enough numeric data
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return {"message": "Not enough numeric columns for predictive modeling."}
    
    # Prepare data - use first numeric column as target for demonstration
    target_col = numeric_cols[0]
    feature_cols = [col for col in numeric_cols if col != target_col]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run PCA
    if X_scaled.shape[1] > 1:
        pca = PCA(n_components=min(3, X_scaled.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        
        results["pca"] = {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "total_explained_variance": sum(pca.explained_variance_ratio_),
            "components": pca.components_.tolist() if X_scaled.shape[1] <= 10 else "Too many features to display"
        }
    
    # Run clustering
    if len(df) > 1:
        kmeans = KMeans(n_clusters=min(3, len(df)), random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Count samples in each cluster
        cluster_counts = np.bincount(clusters)
        
        results["clustering"] = {
            "algorithm": "KMeans",
            "n_clusters": len(cluster_counts),
            "samples_per_cluster": cluster_counts.tolist(),
            "inertia": kmeans.inertia_
        }
    
    # Run a simple predictive model
    try:
        # Check if target is categorical or continuous
        if df[target_col].nunique() < 10:  # Assuming categorical if few unique values
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model_type = "classification"
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model_type = "regression"
        
        # Simple train/test split (70/30)
        split_idx = int(0.7 * len(df))
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        model.fit(X_train, y_train)
        
        # Get feature importances
        importances = model.feature_importances_
        feature_importance = {feature_cols[i]: float(importances[i]) for i in range(len(feature_cols))}
        
        # Sort by importance
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        results["predictive_model"] = {
            "model_type": model_type,
            "target_column": target_col,
            "feature_importance": sorted_importance,
            "top_features": list(sorted_importance.keys())[:3]
        }
        
    except Exception as e:
        results["predictive_model"] = {"error": str(e)}
    
    return results

@app.post("/analyze")
async def analyze_data(file: UploadFile = File(...)):
    """
    Analyze uploaded data file.
    
    This endpoint accepts CSV or Excel files and performs:
    - Data cleaning
    - Exploratory data analysis
    - Basic visualizations
    - Insights generation
    - Simple predictive modeling
    """
    try:
        # Read the file
        df = read_file(file)
        
        # Clean the data
        df_cleaned = clean_data(df)
        
        # Generate analysis results
        results = {
            "summary": get_data_summary(df_cleaned),
            "insights": generate_insights(df_cleaned),
            "visualizations": generate_visualizations(df_cleaned),
            "predictive_modeling": run_predictive_modeling(df_cleaned)
        }
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Data Analytics API is running. Use /analyze endpoint to upload and analyze data."}

if __name__ == "__main__":
    import uvicorn
    # Run the server with CORS enabled
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)