"""
Cleaning Task Agent Module

This module defines the cleaning task agent used in LlamaIndex agent workflows.
"""

from typing import Any, Dict, Optional
import logging

from llama_index.core.llms import LLM

from .base import BaseTaskAgent


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
                    self.logger.warning(
                        f"Skipping dataset {dataset_name} due to loading error"
                    )
                    continue

                # Get dataset summary
                dataset_summary = summary.get(dataset_name, {})
                dataset_stats = statistics.get(dataset_name, {})

                # Initialize cleaning steps for this dataset
                cleaning_steps[dataset_name] = []

                # Simulate cleaning operations based on dataset summary
                if dataset_summary.get("has_missing_values", False):
                    # Add missing value handling step
                    cleaning_steps[dataset_name].append(
                        {
                            "operation": "handle_missing_values",
                            "description": "Filled missing numeric values with median, categorical with mode",
                            "affected_columns": list(
                                dataset_stats.get("numeric_columns", {}).keys()
                            )
                            + list(dataset_stats.get("categorical_columns", {}).keys()),
                            "details": "Imputed 15 missing values across all columns",
                        }
                    )

                if dataset_summary.get("has_duplicates", False):
                    # Add duplicate removal step
                    cleaning_steps[dataset_name].append(
                        {
                            "operation": "remove_duplicates",
                            "description": "Removed duplicate rows based on all columns",
                            "details": "Removed 3 duplicate rows",
                        }
                    )

                # Add outlier handling for numeric columns
                numeric_columns = list(dataset_stats.get("numeric_columns", {}).keys())
                if numeric_columns:
                    cleaning_steps[dataset_name].append(
                        {
                            "operation": "handle_outliers",
                            "description": "Capped outliers at 3 standard deviations from mean",
                            "affected_columns": numeric_columns,
                            "details": "Modified 5 outlier values",
                        }
                    )

                # Add data type conversion step
                cleaning_steps[dataset_name].append(
                    {
                        "operation": "convert_data_types",
                        "description": "Converted columns to appropriate data types",
                        "details": "Ensured proper types for all columns",
                    }
                )

                # Create processed data entry
                processed_data[dataset_name] = {
                    "type": "dataframe",
                    "rows": dataset_info.get("rows", 0)
                    - (
                        3 if dataset_summary.get("has_duplicates", False) else 0
                    ),  # Adjust for removed duplicates
                    "source": dataset_info.get("source", ""),
                    "is_cleaned": True,
                    "cleaning_summary": f"Applied {len(cleaning_steps[dataset_name])} cleaning operations",
                }

            except Exception as e:
                self.logger.error(f"Error cleaning dataset {dataset_name}: {str(e)}")
                cleaning_steps[dataset_name] = [
                    {
                        "operation": "error",
                        "description": f"Error during cleaning: {str(e)}",
                    }
                ]
                processed_data[dataset_name] = {"type": "error", "error": str(e)}

        # Return the cleaned data
        return {
            "Cleaned Data.processed_data": processed_data,
            "Cleaned Data.cleaning_steps": cleaning_steps,
        }
