"""
Visualization Task Agent Module

This module defines the visualization task agent used in LlamaIndex agent workflows.
"""

from typing import Any, Dict, Optional
import logging

from llama_index.core.llms import LLM

from .base import BaseTaskAgent


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
            "theme": "light",
        }

        # Process each dataset
        for dataset_name, dataset_info in processed_data.items():
            try:
                # Skip datasets with errors
                if dataset_info.get("type") == "error":
                    self.logger.warning(
                        f"Skipping dataset {dataset_name} due to visualization error"
                    )
                    continue

                # Get dataset insights and findings
                dataset_insights = insights.get(dataset_name, [])
                dataset_findings = findings.get(dataset_name, {})

                # Initialize plots for this dataset
                plots[dataset_name] = {
                    "data_exploration": [],
                    "analysis": [],
                    "model_performance": [],
                }

                # Create data exploration visualizations
                plots[dataset_name]["data_exploration"] = [
                    {
                        "type": "histogram",
                        "title": "Distribution of column1",
                        "x_axis": "column1",
                        "y_axis": "frequency",
                        "description": "Histogram showing the distribution of values in column1",
                    },
                    {
                        "type": "box_plot",
                        "title": "Box Plot of Numeric Features",
                        "features": ["column1", "column3"],
                        "description": "Box plot showing the distribution of numeric features",
                    },
                    {
                        "type": "bar_chart",
                        "title": "Category Distribution in column2",
                        "x_axis": "category",
                        "y_axis": "count",
                        "description": "Bar chart showing the distribution of categories in column2",
                    },
                ]

                # Create analysis visualizations based on insights
                for insight in dataset_insights:
                    if insight.get("type") == "correlation":
                        plots[dataset_name]["analysis"].append(
                            {
                                "type": "scatter_plot",
                                "title": "Correlation between column1 and column3",
                                "x_axis": "column1",
                                "y_axis": "column3",
                                "trend_line": True,
                                "description": insight.get("description", ""),
                            }
                        )
                    elif insight.get("type") == "time_series":
                        plots[dataset_name]["analysis"].append(
                            {
                                "type": "line_chart",
                                "title": "Time Series Analysis",
                                "x_axis": "date",
                                "y_axis": "value",
                                "description": insight.get("description", ""),
                            }
                        )

                # Create model performance visualizations
                if dataset_name in performance:
                    dataset_performance = performance.get(dataset_name, {})
                    best_model = dataset_performance.get("best_model")

                    if best_model:
                        # Add model comparison chart
                        plots[dataset_name]["model_performance"].append(
                            {
                                "type": "bar_chart",
                                "title": "Model Performance Comparison",
                                "x_axis": "model",
                                "y_axis": "metric_value",
                                "models": list(dataset_performance.keys()),
                                "metrics": (
                                    ["r2", "mse", "mae"]
                                    if "regression" in best_model
                                    else ["accuracy", "precision", "recall", "f1"]
                                ),
                                "description": f"Comparison of performance metrics across different models, with {best_model} performing best",
                            }
                        )

                        # Add feature importance chart for the best model
                        plots[dataset_name]["model_performance"].append(
                            {
                                "type": "bar_chart",
                                "title": f"Feature Importance for {best_model}",
                                "x_axis": "feature",
                                "y_axis": "importance",
                                "features": trained_models.get(dataset_name, {})
                                .get(best_model, {})
                                .get("features", []),
                                "description": "Relative importance of features in the best performing model",
                            }
                        )

                # Add dataset section to dashboard
                dashboard["sections"].append(
                    {
                        "title": f"Analysis of {dataset_name}",
                        "plots": [
                            {
                                "id": "distribution",
                                "title": "Data Distribution",
                                "plot_ref": f"{dataset_name}.data_exploration.0",
                            },
                            {
                                "id": "correlation",
                                "title": "Feature Correlations",
                                "plot_ref": (
                                    f"{dataset_name}.analysis.0"
                                    if plots[dataset_name]["analysis"]
                                    else None
                                ),
                            },
                            {
                                "id": "model_comparison",
                                "title": "Model Comparison",
                                "plot_ref": (
                                    f"{dataset_name}.model_performance.0"
                                    if plots[dataset_name]["model_performance"]
                                    else None
                                ),
                            },
                        ],
                        "summary": dataset_findings.get("summary", ""),
                    }
                )

            except Exception as e:
                self.logger.error(
                    f"Error creating visualizations for dataset {dataset_name}: {str(e)}"
                )
                plots[dataset_name] = {"error": str(e)}

        # Add summary section to dashboard
        dashboard["sections"].append(
            {
                "title": "Executive Summary",
                "content": "This dashboard presents the results of our data analysis and modeling efforts.",
                "key_findings": [
                    finding.get("summary", "")
                    for finding in findings.values()
                    if isinstance(finding, dict) and "error" not in finding
                ],
            }
        )

        # Return the visualizations
        return {"Visualizations.plots": plots, "Visualizations.dashboard": dashboard}
