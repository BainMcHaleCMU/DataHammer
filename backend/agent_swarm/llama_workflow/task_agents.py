"""
Task Agents Module

This module defines the task agents used in LlamaIndex agent workflows.
"""

from typing import Any, Dict, List, Optional
import logging
from abc import ABC, abstractmethod

from llama_index.core.llms import LLM
from llama_index.core.settings import Settings
from llama_index.experimental.agent_workflow import TaskAgent


class BaseTaskAgent(TaskAgent, ABC):
    """
    Base class for all task agents in the workflow.
    
    All specialized task agents inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, name: str, llm: Optional[LLM] = None):
        """
        Initialize the base task agent.
        
        Args:
            name: The name of the agent
            llm: Optional language model to use
        """
        self.name = name
        self.llm = llm or Settings.llm
        if not self.llm:
            raise ValueError("No language model available. Please configure Settings.llm or provide an LLM.")
        
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's primary functionality.
        
        Args:
            input_data: Input data for the task
            
        Returns:
            Dict containing results and any suggestions for next steps
        """
        pass


class DataLoadingTaskAgent(BaseTaskAgent):
    """
    Task agent for loading data from various sources.
    
    Responsibilities:
    - Loading data from files (CSV, Excel, JSON, etc.)
    - Loading data from databases
    - Loading data from APIs
    - Basic data validation
    """
    
    def __init__(self, llm: Optional[LLM] = None):
        """Initialize the DataLoadingTaskAgent."""
        super().__init__(name="DataLoadingAgent", llm=llm)
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the data loading task.
        
        Args:
            input_data: Input data for the task
            
        Returns:
            Dict containing loaded data and metadata
        """
        environment = input_data.get("environment", {})
        goals = input_data.get("goals", [])
        
        # Get data sources from the environment
        data_sources = environment.get("Data", {})
        specific_sources = input_data.get("data_sources", {})
        
        # If specific sources are provided, use those instead
        if specific_sources:
            data_sources = specific_sources
        
        self.logger.info(f"Loading data from {len(data_sources)} sources")
        
        loaded_data = {}
        schema_info = {}
        
        # Process each data source
        for source_name, source_info in data_sources.items():
            try:
                source_type = source_info.get("type", "unknown")
                source_path = source_info.get("path", "")
                
                if source_type == "csv":
                    # Simulate loading CSV data
                    self.logger.info(f"Loading CSV data from {source_path}")
                    loaded_data[source_name] = {
                        "type": "dataframe",
                        "rows": 100,  # Simulated row count
                        "source": source_path
                    }
                    schema_info[source_name] = {
                        "columns": ["column1", "column2", "column3"],
                        "types": ["int", "string", "float"]
                    }
                elif source_type == "excel":
                    # Simulate loading Excel data
                    self.logger.info(f"Loading Excel data from {source_path}")
                    loaded_data[source_name] = {
                        "type": "dataframe",
                        "rows": 200,  # Simulated row count
                        "source": source_path
                    }
                    schema_info[source_name] = {
                        "columns": ["column1", "column2", "column3"],
                        "types": ["int", "string", "float"]
                    }
                elif source_type == "database":
                    # Simulate loading database data
                    connection_string = source_info.get("connection", "")
                    query = source_info.get("query", "")
                    self.logger.info(f"Loading database data from {connection_string}")
                    loaded_data[source_name] = {
                        "type": "dataframe",
                        "rows": 500,  # Simulated row count
                        "source": f"DB: {connection_string}"
                    }
                    schema_info[source_name] = {
                        "columns": ["id", "name", "value", "date"],
                        "types": ["int", "string", "float", "datetime"]
                    }
                else:
                    self.logger.warning(f"Unknown data source type: {source_type}")
                    loaded_data[source_name] = {
                        "type": "unknown",
                        "error": f"Unsupported data source type: {source_type}"
                    }
            except Exception as e:
                self.logger.error(f"Error loading data from {source_name}: {str(e)}")
                loaded_data[source_name] = {
                    "type": "error",
                    "error": str(e)
                }
        
        # Return the loaded data
        return {
            "Data Overview.raw_data": loaded_data,
            "Data Overview.schema": schema_info
        }


class ExplorationTaskAgent(BaseTaskAgent):
    """
    Task agent for exploring and understanding data.
    
    Responsibilities:
    - Data profiling
    - Statistical analysis
    - Feature discovery
    - Correlation analysis
    """
    
    def __init__(self, llm: Optional[LLM] = None):
        """Initialize the ExplorationTaskAgent."""
        super().__init__(name="ExplorationAgent", llm=llm)
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the data exploration task.
        
        Args:
            input_data: Input data for the task
            
        Returns:
            Dict containing exploration results
        """
        environment = input_data.get("environment", {})
        goals = input_data.get("goals", [])
        
        # Get raw data from the environment
        data_overview = environment.get("Data Overview", {})
        raw_data = data_overview.get("raw_data", {})
        schema = data_overview.get("schema", {})
        
        self.logger.info("Exploring data")
        
        # Initialize results
        summary = {}
        statistics = {}
        
        # Process each dataset
        for dataset_name, dataset_info in raw_data.items():
            try:
                # Skip datasets with errors
                if dataset_info.get("type") == "error":
                    self.logger.warning(f"Skipping dataset {dataset_name} due to loading error")
                    continue
                
                # Get dataset schema
                dataset_schema = schema.get(dataset_name, {})
                columns = dataset_schema.get("columns", [])
                
                # Generate summary for this dataset
                summary[dataset_name] = {
                    "row_count": dataset_info.get("rows", 0),
                    "column_count": len(columns),
                    "columns": columns,
                    "completeness": "95%",  # Simulated completeness
                    "has_missing_values": True,  # Simulated missing values check
                    "has_duplicates": False,  # Simulated duplicates check
                }
                
                # Generate statistics for this dataset
                statistics[dataset_name] = {
                    "numeric_columns": {},
                    "categorical_columns": {},
                    "datetime_columns": {}
                }
                
                # Simulate statistics for each column based on its type
                for i, col in enumerate(columns):
                    col_type = dataset_schema.get("types", [])[i] if i < len(dataset_schema.get("types", [])) else "unknown"
                    
                    if col_type in ["int", "float", "double", "numeric"]:
                        # Numeric column statistics
                        statistics[dataset_name]["numeric_columns"][col] = {
                            "min": 0,  # Simulated min
                            "max": 100,  # Simulated max
                            "mean": 50,  # Simulated mean
                            "median": 48,  # Simulated median
                            "std": 15,  # Simulated standard deviation
                            "missing": 2  # Simulated missing count
                        }
                    elif col_type in ["string", "text", "varchar"]:
                        # Categorical column statistics
                        statistics[dataset_name]["categorical_columns"][col] = {
                            "unique_values": 10,  # Simulated unique count
                            "most_common": ["value1", "value2", "value3"],  # Simulated most common values
                            "missing": 1  # Simulated missing count
                        }
                    elif col_type in ["date", "datetime", "timestamp"]:
                        # Datetime column statistics
                        statistics[dataset_name]["datetime_columns"][col] = {
                            "min_date": "2020-01-01",  # Simulated min date
                            "max_date": "2023-12-31",  # Simulated max date
                            "missing": 0  # Simulated missing count
                        }
            except Exception as e:
                self.logger.error(f"Error exploring dataset {dataset_name}: {str(e)}")
                summary[dataset_name] = {
                    "error": str(e)
                }
        
        # Return the exploration results
        return {
            "Data Overview.summary": summary,
            "Data Overview.statistics": statistics
        }


class CleaningTaskAgent(BaseTaskAgent):
    """
    Task agent for cleaning and preprocessing data.
    
    Responsibilities:
    - Handling missing values
    - Removing duplicates
    - Handling outliers
    - Feature engineering
    - Data transformation
    """
    
    def __init__(self, llm: Optional[LLM] = None):
        """Initialize the CleaningTaskAgent."""
        super().__init__(name="CleaningAgent", llm=llm)
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the data cleaning task.
        
        Args:
            input_data: Input data for the task
            
        Returns:
            Dict containing cleaned data
        """
        environment = input_data.get("environment", {})
        goals = input_data.get("goals", [])
        
        # Get raw data and summary from the environment
        data_overview = environment.get("Data Overview", {})
        raw_data = data_overview.get("raw_data", {})
        summary = data_overview.get("summary", {})
        statistics = data_overview.get("statistics", {})
        
        self.logger.info("Cleaning data")
        
        # Initialize results
        processed_data = {}
        cleaning_steps = {}
        
        # Process each dataset
        for dataset_name, dataset_info in raw_data.items():
            try:
                # Skip datasets with errors
                if dataset_info.get("type") == "error":
                    self.logger.warning(f"Skipping dataset {dataset_name} due to loading error")
                    continue
                
                # Get dataset summary
                dataset_summary = summary.get(dataset_name, {})
                dataset_stats = statistics.get(dataset_name, {})
                
                # Initialize cleaning steps for this dataset
                cleaning_steps[dataset_name] = []
                
                # Simulate cleaning operations based on dataset summary
                if dataset_summary.get("has_missing_values", False):
                    # Add missing value handling step
                    cleaning_steps[dataset_name].append({
                        "operation": "handle_missing_values",
                        "description": "Filled missing numeric values with median, categorical with mode",
                        "affected_columns": list(dataset_stats.get("numeric_columns", {}).keys()) + 
                                           list(dataset_stats.get("categorical_columns", {}).keys()),
                        "details": "Imputed 15 missing values across all columns"
                    })
                
                if dataset_summary.get("has_duplicates", False):
                    # Add duplicate removal step
                    cleaning_steps[dataset_name].append({
                        "operation": "remove_duplicates",
                        "description": "Removed duplicate rows based on all columns",
                        "details": "Removed 3 duplicate rows"
                    })
                
                # Add outlier handling for numeric columns
                numeric_columns = list(dataset_stats.get("numeric_columns", {}).keys())
                if numeric_columns:
                    cleaning_steps[dataset_name].append({
                        "operation": "handle_outliers",
                        "description": "Capped outliers at 3 standard deviations from mean",
                        "affected_columns": numeric_columns,
                        "details": "Modified 5 outlier values"
                    })
                
                # Add data type conversion step
                cleaning_steps[dataset_name].append({
                    "operation": "convert_data_types",
                    "description": "Converted columns to appropriate data types",
                    "details": "Ensured proper types for all columns"
                })
                
                # Create processed data entry
                processed_data[dataset_name] = {
                    "type": "dataframe",
                    "rows": dataset_info.get("rows", 0) - 
                           (3 if dataset_summary.get("has_duplicates", False) else 0),  # Adjust for removed duplicates
                    "source": dataset_info.get("source", ""),
                    "is_cleaned": True,
                    "cleaning_summary": f"Applied {len(cleaning_steps[dataset_name])} cleaning operations"
                }
                
            except Exception as e:
                self.logger.error(f"Error cleaning dataset {dataset_name}: {str(e)}")
                cleaning_steps[dataset_name] = [{
                    "operation": "error",
                    "description": f"Error during cleaning: {str(e)}"
                }]
                processed_data[dataset_name] = {
                    "type": "error",
                    "error": str(e)
                }
        
        # Return the cleaned data
        return {
            "Cleaned Data.processed_data": processed_data,
            "Cleaned Data.cleaning_steps": cleaning_steps
        }


class AnalysisTaskAgent(BaseTaskAgent):
    """
    Task agent for in-depth data analysis.
    
    Responsibilities:
    - Advanced statistical analysis
    - Hypothesis testing
    - Segmentation analysis
    - Time series analysis
    - Pattern discovery
    """
    
    def __init__(self, llm: Optional[LLM] = None):
        """Initialize the AnalysisTaskAgent."""
        super().__init__(name="AnalysisAgent", llm=llm)
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the data analysis task.
        
        Args:
            input_data: Input data for the task
            
        Returns:
            Dict containing analysis results
        """
        environment = input_data.get("environment", {})
        goals = input_data.get("goals", [])
        
        # Get cleaned data from the environment
        cleaned_data = environment.get("Cleaned Data", {})
        processed_data = cleaned_data.get("processed_data", {})
        cleaning_steps = cleaned_data.get("cleaning_steps", {})
        
        # Get data overview information
        data_overview = environment.get("Data Overview", {})
        statistics = data_overview.get("statistics", {})
        
        self.logger.info("Analyzing data")
        
        # Initialize results
        insights = {}
        findings = {}
        
        # Process each dataset
        for dataset_name, dataset_info in processed_data.items():
            try:
                # Skip datasets with errors
                if dataset_info.get("type") == "error":
                    self.logger.warning(f"Skipping dataset {dataset_name} due to processing error")
                    continue
                
                # Get dataset statistics
                dataset_stats = statistics.get(dataset_name, {})
                
                # Generate insights for this dataset
                insights[dataset_name] = []
                
                # Add general dataset insights
                insights[dataset_name].append({
                    "type": "general",
                    "description": f"Dataset contains {dataset_info.get('rows', 0)} rows after cleaning",
                    "importance": "medium"
                })
                
                # Add insights about numeric columns
                numeric_columns = dataset_stats.get("numeric_columns", {})
                if numeric_columns:
                    # Simulate correlation analysis
                    insights[dataset_name].append({
                        "type": "correlation",
                        "description": "Strong positive correlation detected between column1 and column3",
                        "details": "Correlation coefficient: 0.85",
                        "importance": "high"
                    })
                    
                    # Simulate distribution analysis
                    insights[dataset_name].append({
                        "type": "distribution",
                        "description": "column1 shows a normal distribution with slight right skew",
                        "details": "Skewness: 0.32",
                        "importance": "medium"
                    })
                
                # Add insights about categorical columns
                categorical_columns = dataset_stats.get("categorical_columns", {})
                if categorical_columns:
                    # Simulate categorical analysis
                    insights[dataset_name].append({
                        "type": "categorical",
                        "description": "Significant imbalance detected in column2 categories",
                        "details": "80% of values belong to a single category",
                        "importance": "high"
                    })
                
                # Add insights about datetime columns
                datetime_columns = dataset_stats.get("datetime_columns", {})
                if datetime_columns:
                    # Simulate time series analysis
                    insights[dataset_name].append({
                        "type": "time_series",
                        "description": "Seasonal pattern detected in the data",
                        "details": "Quarterly peaks observed",
                        "importance": "high"
                    })
                
                # Generate findings for this dataset
                findings[dataset_name] = {
                    "summary": f"Analysis of {dataset_name} revealed {len(insights[dataset_name])} key insights",
                    "key_variables": list(numeric_columns.keys())[:2] if numeric_columns else [],
                    "potential_issues": [
                        "Data imbalance in categorical variables",
                        "Some outliers still present after cleaning"
                    ] if categorical_columns else [],
                    "recommendations": [
                        "Consider feature engineering to create interaction terms",
                        "Normalize numeric features before modeling",
                        "Consider stratified sampling for model training"
                    ]
                }
                
            except Exception as e:
                self.logger.error(f"Error analyzing dataset {dataset_name}: {str(e)}")
                insights[dataset_name] = [{
                    "type": "error",
                    "description": f"Error during analysis: {str(e)}",
                    "importance": "high"
                }]
                findings[dataset_name] = {
                    "error": str(e)
                }
        
        # Return the analysis results
        return {
            "Analysis Results.insights": insights,
            "Analysis Results.findings": findings
        }


class ModelingTaskAgent(BaseTaskAgent):
    """
    Task agent for building and evaluating predictive models.
    
    Responsibilities:
    - Feature selection
    - Model selection
    - Model training
    - Model evaluation
    - Hyperparameter tuning
    """
    
    def __init__(self, llm: Optional[LLM] = None):
        """Initialize the ModelingTaskAgent."""
        super().__init__(name="ModelingAgent", llm=llm)
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the modeling task.
        
        Args:
            input_data: Input data for the task
            
        Returns:
            Dict containing model and performance metrics
        """
        environment = input_data.get("environment", {})
        goals = input_data.get("goals", [])
        
        # Get cleaned data from the environment
        cleaned_data = environment.get("Cleaned Data", {})
        processed_data = cleaned_data.get("processed_data", {})
        
        # Get analysis results
        analysis_results = environment.get("Analysis Results", {})
        findings = analysis_results.get("findings", {})
        
        self.logger.info("Building models")
        
        # Initialize results
        trained_models = {}
        performance_metrics = {}
        
        # Process each dataset
        for dataset_name, dataset_info in processed_data.items():
            try:
                # Skip datasets with errors
                if dataset_info.get("type") == "error":
                    self.logger.warning(f"Skipping dataset {dataset_name} due to processing error")
                    continue
                
                # Get dataset findings
                dataset_findings = findings.get(dataset_name, {})
                
                # Determine model type based on goals and data
                # For simplicity, we'll simulate different model types
                model_types = ["linear_regression", "random_forest", "gradient_boosting"]
                
                # Train multiple models for each dataset
                trained_models[dataset_name] = {}
                performance_metrics[dataset_name] = {}
                
                for model_type in model_types:
                    # Simulate model training
                    self.logger.info(f"Training {model_type} model for {dataset_name}")
                    
                    # Create model metadata
                    trained_models[dataset_name][model_type] = {
                        "type": model_type,
                        "features": dataset_findings.get("key_variables", []) + ["additional_feature"],
                        "target": "target_variable",  # Simulated target variable
                        "hyperparameters": self._get_default_hyperparameters(model_type),
                        "training_time": "2.5 seconds",  # Simulated training time
                        "timestamp": "2023-04-13T10:30:00Z"  # Simulated timestamp
                    }
                    
                    # Create performance metrics
                    performance_metrics[dataset_name][model_type] = self._generate_performance_metrics(model_type)
                
                # Determine best model
                best_model = max(
                    performance_metrics[dataset_name].items(),
                    key=lambda x: x[1].get("metrics", {}).get("r2", 0)
                    if "regression" in x[0] else x[1].get("metrics", {}).get("accuracy", 0)
                )[0]
                
                performance_metrics[dataset_name]["best_model"] = best_model
                
            except Exception as e:
                self.logger.error(f"Error modeling dataset {dataset_name}: {str(e)}")
                trained_models[dataset_name] = {
                    "error": str(e)
                }
                performance_metrics[dataset_name] = {
                    "error": str(e)
                }
        
        # Return the modeling results
        return {
            "Models.trained_model": trained_models,
            "Models.performance": performance_metrics
        }
    
    def _get_default_hyperparameters(self, model_type: str) -> Dict[str, Any]:
        """Get default hyperparameters for a model type."""
        if model_type == "linear_regression":
            return {
                "fit_intercept": True,
                "normalize": False
            }
        elif model_type == "random_forest":
            return {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2,
                "min_samples_leaf": 1
            }
        elif model_type == "gradient_boosting":
            return {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "subsample": 0.8
            }
        else:
            return {}
    
    def _generate_performance_metrics(self, model_type: str) -> Dict[str, Any]:
        """Generate simulated performance metrics for a model."""
        if "regression" in model_type:
            return {
                "metrics": {
                    "r2": 0.85 if model_type == "gradient_boosting" else (0.82 if model_type == "random_forest" else 0.75),
                    "mse": 0.15 if model_type == "gradient_boosting" else (0.18 if model_type == "random_forest" else 0.25),
                    "mae": 0.12 if model_type == "gradient_boosting" else (0.15 if model_type == "random_forest" else 0.20)
                },
                "cross_validation": {
                    "method": "5-fold",
                    "r2_scores": [0.83, 0.86, 0.84, 0.87, 0.85] if model_type == "gradient_boosting" else 
                                 ([0.80, 0.83, 0.81, 0.84, 0.82] if model_type == "random_forest" else 
                                  [0.73, 0.76, 0.74, 0.77, 0.75])
                }
            }
        else:
            return {
                "metrics": {
                    "accuracy": 0.92 if model_type == "gradient_boosting" else (0.90 if model_type == "random_forest" else 0.85),
                    "precision": 0.91 if model_type == "gradient_boosting" else (0.89 if model_type == "random_forest" else 0.84),
                    "recall": 0.90 if model_type == "gradient_boosting" else (0.88 if model_type == "random_forest" else 0.83),
                    "f1": 0.905 if model_type == "gradient_boosting" else (0.885 if model_type == "random_forest" else 0.835)
                },
                "cross_validation": {
                    "method": "5-fold",
                    "accuracy_scores": [0.91, 0.93, 0.92, 0.94, 0.92] if model_type == "gradient_boosting" else 
                                      ([0.89, 0.91, 0.90, 0.92, 0.90] if model_type == "random_forest" else 
                                       [0.84, 0.86, 0.85, 0.87, 0.85])
                }
            }


class VisualizationTaskAgent(BaseTaskAgent):
    """
    Task agent for creating data visualizations.
    
    Responsibilities:
    - Creating charts and graphs
    - Creating interactive visualizations
    - Creating dashboards
    - Visualizing model results
    """
    
    def __init__(self, llm: Optional[LLM] = None):
        """Initialize the VisualizationTaskAgent."""
        super().__init__(name="VisualizationAgent", llm=llm)
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the visualization task.
        
        Args:
            input_data: Input data for the task
            
        Returns:
            Dict containing visualizations
        """
        environment = input_data.get("environment", {})
        goals = input_data.get("goals", [])
        
        # Get data from the environment
        cleaned_data = environment.get("Cleaned Data", {})
        processed_data = cleaned_data.get("processed_data", {})
        
        analysis_results = environment.get("Analysis Results", {})
        insights = analysis_results.get("insights", {})
        findings = analysis_results.get("findings", {})
        
        models = environment.get("Models", {})
        trained_models = models.get("trained_model", {})
        performance = models.get("performance", {})
        
        self.logger.info("Creating visualizations")
        
        # Initialize results
        plots = {}
        dashboard = {
            "title": "Data Analysis Dashboard",
            "sections": [],
            "layout": "grid",
            "theme": "light"
        }
        
        # Process each dataset
        for dataset_name, dataset_info in processed_data.items():
            try:
                # Skip datasets with errors
                if dataset_info.get("type") == "error":
                    self.logger.warning(f"Skipping dataset {dataset_name} due to visualization error")
                    continue
                
                # Get dataset insights and findings
                dataset_insights = insights.get(dataset_name, [])
                dataset_findings = findings.get(dataset_name, {})
                
                # Initialize plots for this dataset
                plots[dataset_name] = {
                    "data_exploration": [],
                    "analysis": [],
                    "model_performance": []
                }
                
                # Create data exploration visualizations
                plots[dataset_name]["data_exploration"] = [
                    {
                        "type": "histogram",
                        "title": "Distribution of column1",
                        "x_axis": "column1",
                        "y_axis": "frequency",
                        "description": "Histogram showing the distribution of values in column1"
                    },
                    {
                        "type": "box_plot",
                        "title": "Box Plot of Numeric Features",
                        "features": ["column1", "column3"],
                        "description": "Box plot showing the distribution of numeric features"
                    },
                    {
                        "type": "bar_chart",
                        "title": "Category Distribution in column2",
                        "x_axis": "category",
                        "y_axis": "count",
                        "description": "Bar chart showing the distribution of categories in column2"
                    }
                ]
                
                # Create analysis visualizations based on insights
                for insight in dataset_insights:
                    if insight.get("type") == "correlation":
                        plots[dataset_name]["analysis"].append({
                            "type": "scatter_plot",
                            "title": "Correlation between column1 and column3",
                            "x_axis": "column1",
                            "y_axis": "column3",
                            "trend_line": True,
                            "description": insight.get("description", "")
                        })
                    elif insight.get("type") == "time_series":
                        plots[dataset_name]["analysis"].append({
                            "type": "line_chart",
                            "title": "Time Series Analysis",
                            "x_axis": "date",
                            "y_axis": "value",
                            "description": insight.get("description", "")
                        })
                
                # Create model performance visualizations
                if dataset_name in performance:
                    dataset_performance = performance.get(dataset_name, {})
                    best_model = dataset_performance.get("best_model")
                    
                    if best_model:
                        # Add model comparison chart
                        plots[dataset_name]["model_performance"].append({
                            "type": "bar_chart",
                            "title": "Model Performance Comparison",
                            "x_axis": "model",
                            "y_axis": "metric_value",
                            "models": list(dataset_performance.keys()),
                            "metrics": ["r2", "mse", "mae"] if "regression" in best_model else ["accuracy", "precision", "recall", "f1"],
                            "description": f"Comparison of performance metrics across different models, with {best_model} performing best"
                        })
                        
                        # Add feature importance chart for the best model
                        plots[dataset_name]["model_performance"].append({
                            "type": "bar_chart",
                            "title": f"Feature Importance for {best_model}",
                            "x_axis": "feature",
                            "y_axis": "importance",
                            "features": trained_models.get(dataset_name, {}).get(best_model, {}).get("features", []),
                            "description": "Relative importance of features in the best performing model"
                        })
                
                # Add dataset section to dashboard
                dashboard["sections"].append({
                    "title": f"Analysis of {dataset_name}",
                    "plots": [
                        {"id": "distribution", "title": "Data Distribution", "plot_ref": f"{dataset_name}.data_exploration.0"},
                        {"id": "correlation", "title": "Feature Correlations", "plot_ref": f"{dataset_name}.analysis.0" if plots[dataset_name]["analysis"] else None},
                        {"id": "model_comparison", "title": "Model Comparison", "plot_ref": f"{dataset_name}.model_performance.0" if plots[dataset_name]["model_performance"] else None}
                    ],
                    "summary": dataset_findings.get("summary", "")
                })
                
            except Exception as e:
                self.logger.error(f"Error creating visualizations for dataset {dataset_name}: {str(e)}")
                plots[dataset_name] = {
                    "error": str(e)
                }
        
        # Add summary section to dashboard
        dashboard["sections"].append({
            "title": "Executive Summary",
            "content": "This dashboard presents the results of our data analysis and modeling efforts.",
            "key_findings": [finding.get("summary", "") for finding in findings.values() if isinstance(finding, dict) and "error" not in finding]
        })
        
        # Return the visualizations
        return {
            "Visualizations.plots": plots,
            "Visualizations.dashboard": dashboard
        }


class ReportingTaskAgent(BaseTaskAgent):
    """
    Task agent for generating reports and documentation.
    
    Responsibilities:
    - Creating summary reports
    - Documenting findings
    - Creating presentations
    - Generating recommendations
    """
    
    def __init__(self, llm: Optional[LLM] = None):
        """Initialize the ReportingTaskAgent."""
        super().__init__(name="ReportingAgent", llm=llm)
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the reporting task.
        
        Args:
            input_data: Input data for the task
            
        Returns:
            Dict containing report and documentation
        """
        environment = input_data.get("environment", {})
        goals = input_data.get("goals", [])
        
        # Get data from the environment
        data_overview = environment.get("Data Overview", {})
        summary = data_overview.get("summary", {})
        
        cleaned_data = environment.get("Cleaned Data", {})
        cleaning_steps = cleaned_data.get("cleaning_steps", {})
        
        analysis_results = environment.get("Analysis Results", {})
        insights = analysis_results.get("insights", {})
        findings = analysis_results.get("findings", {})
        
        models = environment.get("Models", {})
        trained_models = models.get("trained_model", {})
        performance = models.get("performance", {})
        
        visualizations = environment.get("Visualizations", {})
        plots = visualizations.get("plots", {})
        dashboard = visualizations.get("dashboard", {})
        
        self.logger.info("Generating report")
        
        # Create Jupyter notebook cells
        jupyter_cells = []
        
        # Add title and introduction
        jupyter_cells.extend([
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "# Data Analysis Report\n\n## Introduction\n\nThis notebook contains the results of our data analysis and modeling efforts."
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "## Goals\n\n" + "\n".join([f"- {goal}" for goal in goals])
            }
        ])
        
        # Add data overview section
        jupyter_cells.extend([
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "## Data Overview\n\nThis section provides an overview of the datasets used in the analysis."
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": "# Code to display data overview\nimport pandas as pd\nimport json\n\n# Display summary of datasets\nprint('Dataset Summary:')\nprint(json.dumps(" + str(summary) + ", indent=2))",
                "execution_count": None,
                "outputs": []
            }
        ])
        
        # Add data cleaning section
        jupyter_cells.extend([
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "## Data Cleaning\n\nThis section describes the data cleaning steps performed."
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": "# Code to display cleaning steps\nimport pandas as pd\n\n# Display cleaning steps for each dataset\nfor dataset_name, steps in " + str(cleaning_steps) + ".items():\n    print(f'Cleaning steps for {dataset_name}:')\n    for i, step in enumerate(steps):\n        print(f\"  {i+1}. {step['operation']}: {step['description']}\")\n    print()",
                "execution_count": None,
                "outputs": []
            }
        ])
        
        # Add analysis section
        jupyter_cells.extend([
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "## Data Analysis\n\nThis section presents the key insights from our analysis."
            }
        ])
        
        # Add insights for each dataset
        for dataset_name, dataset_insights in insights.items():
            if isinstance(dataset_insights, list):
                jupyter_cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": f"### Insights for {dataset_name}\n\n" + 
                              "\n".join([f"- **{insight.get('type', 'Insight')}**: {insight.get('description', '')}" +
                                        (f" ({insight.get('details', '')})" if insight.get('details') else "")
                                        for insight in dataset_insights if isinstance(insight, dict)])
                })
        
        # Add modeling section
        jupyter_cells.extend([
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "## Modeling Results\n\nThis section presents the results of our modeling efforts."
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": "# Code to display model performance\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Display performance metrics for each dataset and model\nfor dataset_name, dataset_perf in " + str(performance) + ".items():\n    if isinstance(dataset_perf, dict) and 'best_model' in dataset_perf:\n        print(f'Model performance for {dataset_name}:')\n        print(f\"Best model: {dataset_perf['best_model']}\")\n        \n        # Create a simple bar chart of model performance\n        best_model = dataset_perf['best_model']\n        if best_model in dataset_perf and 'metrics' in dataset_perf[best_model]:\n            metrics = dataset_perf[best_model]['metrics']\n            print(f\"Performance metrics: {metrics}\")\n    print()",
                "execution_count": None,
                "outputs": []
            }
        ])
        
        # Add visualization section
        jupyter_cells.extend([
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "## Visualizations\n\nThis section contains key visualizations from our analysis."
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "### Dashboard Overview\n\n" +
                          f"The dashboard contains {len(dashboard.get('sections', []))} sections with various visualizations."
            }
        ])
        
        # Add recommendations section
        recommendations = []
        for dataset_name, finding in findings.items():
            if isinstance(finding, dict) and "recommendations" in finding:
                for rec in finding.get("recommendations", []):
                    recommendations.append(f"- **{dataset_name}**: {rec}")
        
        jupyter_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Recommendations\n\nBased on our analysis, we recommend the following actions:\n\n" +
                      "\n".join(recommendations)
        })
        
        # Create summary
        report_summary = {
            "title": "Data Analysis Report",
            "date": "2023-04-13",
            "datasets_analyzed": list(summary.keys()),
            "key_findings": []
        }
        
        # Add key findings from each dataset
        for dataset_name, finding in findings.items():
            if isinstance(finding, dict) and "summary" in finding:
                report_summary["key_findings"].append({
                    "dataset": dataset_name,
                    "summary": finding.get("summary", ""),
                    "key_variables": finding.get("key_variables", [])
                })
        
        # Create recommendations
        report_recommendations = {
            "data_quality": [
                "Implement data validation checks to prevent missing values",
                "Standardize data collection processes to ensure consistency"
            ],
            "analysis": [
                "Conduct further analysis on correlations between key variables",
                "Investigate seasonal patterns in time series data"
            ],
            "modeling": [
                "Deploy the best performing model in a production environment",
                "Regularly retrain models with new data to maintain accuracy",
                "Consider ensemble methods to improve prediction performance"
            ],
            "business_actions": [
                "Use insights to inform strategic decision making",
                "Develop a data-driven approach to problem solving",
                "Invest in data infrastructure to support ongoing analysis"
            ]
        }
        
        # Return the report
        return {
            "JupyterLogbook": {
                "cells": jupyter_cells,
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3"
                    },
                    "language_info": {
                        "codemirror_mode": {
                            "name": "ipython",
                            "version": 3
                        },
                        "file_extension": ".py",
                        "mimetype": "text/x-python",
                        "name": "python",
                        "nbconvert_exporter": "python",
                        "pygments_lexer": "ipython3",
                        "version": "3.8.10"
                    }
                },
                "nbformat": 4,
                "nbformat_minor": 5
            },
            "Report.summary": report_summary,
            "Report.recommendations": report_recommendations
        }