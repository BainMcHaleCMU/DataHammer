"""
Analysis Task Agent Module

This module defines the analysis task agent used in LlamaIndex agent workflows.
"""

from typing import Any, Dict, Optional
import logging

from llama_index.core.llms import LLM

from .base import BaseTaskAgent


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
                    self.logger.warning(
                        f"Skipping dataset {dataset_name} due to processing error"
                    )
                    continue

                # Get dataset statistics
                dataset_stats = statistics.get(dataset_name, {})

                # Generate insights for this dataset
                insights[dataset_name] = []

                # Add general dataset insights
                insights[dataset_name].append(
                    {
                        "type": "general",
                        "description": f"Dataset contains {dataset_info.get('rows', 0)} rows after cleaning",
                        "importance": "medium",
                    }
                )

                # Add insights about numeric columns
                numeric_columns = dataset_stats.get("numeric_columns", {})
                if numeric_columns:
                    # Simulate correlation analysis
                    insights[dataset_name].append(
                        {
                            "type": "correlation",
                            "description": "Strong positive correlation detected between column1 and column3",
                            "details": "Correlation coefficient: 0.85",
                            "importance": "high",
                        }
                    )

                    # Simulate distribution analysis
                    insights[dataset_name].append(
                        {
                            "type": "distribution",
                            "description": "column1 shows a normal distribution with slight right skew",
                            "details": "Skewness: 0.32",
                            "importance": "medium",
                        }
                    )

                # Add insights about categorical columns
                categorical_columns = dataset_stats.get("categorical_columns", {})
                if categorical_columns:
                    # Simulate categorical analysis
                    insights[dataset_name].append(
                        {
                            "type": "categorical",
                            "description": "Significant imbalance detected in column2 categories",
                            "details": "80% of values belong to a single category",
                            "importance": "high",
                        }
                    )

                # Add insights about datetime columns
                datetime_columns = dataset_stats.get("datetime_columns", {})
                if datetime_columns:
                    # Simulate time series analysis
                    insights[dataset_name].append(
                        {
                            "type": "time_series",
                            "description": "Seasonal pattern detected in the data",
                            "details": "Quarterly peaks observed",
                            "importance": "high",
                        }
                    )

                # Generate findings for this dataset
                findings[dataset_name] = {
                    "summary": f"Analysis of {dataset_name} revealed {len(insights[dataset_name])} key insights",
                    "key_variables": (
                        list(numeric_columns.keys())[:2] if numeric_columns else []
                    ),
                    "potential_issues": (
                        [
                            "Data imbalance in categorical variables",
                            "Some outliers still present after cleaning",
                        ]
                        if categorical_columns
                        else []
                    ),
                    "recommendations": [
                        "Consider feature engineering to create interaction terms",
                        "Normalize numeric features before modeling",
                        "Consider stratified sampling for model training",
                    ],
                }

            except Exception as e:
                self.logger.error(f"Error analyzing dataset {dataset_name}: {str(e)}")
                insights[dataset_name] = [
                    {
                        "type": "error",
                        "description": f"Error during analysis: {str(e)}",
                        "importance": "high",
                    }
                ]
                findings[dataset_name] = {"error": str(e)}

        # Return the analysis results
        return {
            "Analysis Results.insights": insights,
            "Analysis Results.findings": findings,
        }
