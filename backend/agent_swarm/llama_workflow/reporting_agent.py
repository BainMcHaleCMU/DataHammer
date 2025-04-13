"""
Reporting Task Agent Module

This module defines the reporting task agent used in LlamaIndex agent workflows.
"""

from typing import Any, Dict, Optional
import logging

from llama_index.core.llms import LLM

from .base import BaseTaskAgent


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
        jupyter_cells.extend(
            [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": "# Data Analysis Report\n\n## Introduction\n\nThis notebook contains the results of our data analysis and modeling efforts.",
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": "## Goals\n\n"
                    + "\n".join([f"- {goal}" for goal in goals]),
                },
            ]
        )

        # Add data overview section
        jupyter_cells.extend(
            [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": "## Data Overview\n\nThis section provides an overview of the datasets used in the analysis.",
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": "# Code to display data overview\nimport pandas as pd\nimport json\n\n# Display summary of datasets\nprint('Dataset Summary:')\nprint(json.dumps("
                    + str(summary)
                    + ", indent=2))",
                    "execution_count": None,
                    "outputs": [],
                },
            ]
        )

        # Add data cleaning section
        jupyter_cells.extend(
            [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": "## Data Cleaning\n\nThis section describes the data cleaning steps performed.",
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": "# Code to display cleaning steps\nimport pandas as pd\n\n# Display cleaning steps for each dataset\nfor dataset_name, steps in "
                    + str(cleaning_steps)
                    + ".items():\n    print(f'Cleaning steps for {dataset_name}:')\n    for i, step in enumerate(steps):\n        print(f\"  {i+1}. {step['operation']}: {step['description']}\")\n    print()",
                    "execution_count": None,
                    "outputs": [],
                },
            ]
        )

        # Add analysis section
        jupyter_cells.extend(
            [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": "## Data Analysis\n\nThis section presents the key insights from our analysis.",
                }
            ]
        )

        # Add insights for each dataset
        for dataset_name, dataset_insights in insights.items():
            if isinstance(dataset_insights, list):
                jupyter_cells.append(
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": f"### Insights for {dataset_name}\n\n"
                        + "\n".join(
                            [
                                f"- **{insight.get('type', 'Insight')}**: {insight.get('description', '')}"
                                + (
                                    f" ({insight.get('details', '')})"
                                    if insight.get("details")
                                    else ""
                                )
                                for insight in dataset_insights
                                if isinstance(insight, dict)
                            ]
                        ),
                    }
                )

        # Add modeling section
        jupyter_cells.extend(
            [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": "## Modeling Results\n\nThis section presents the results of our modeling efforts.",
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": "# Code to display model performance\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Display performance metrics for each dataset and model\nfor dataset_name, dataset_perf in "
                    + str(performance)
                    + ".items():\n    if isinstance(dataset_perf, dict) and 'best_model' in dataset_perf:\n        print(f'Model performance for {dataset_name}:')\n        print(f\"Best model: {dataset_perf['best_model']}\")\n        \n        # Create a simple bar chart of model performance\n        best_model = dataset_perf['best_model']\n        if best_model in dataset_perf and 'metrics' in dataset_perf[best_model]:\n            metrics = dataset_perf[best_model]['metrics']\n            print(f\"Performance metrics: {metrics}\")\n    print()",
                    "execution_count": None,
                    "outputs": [],
                },
            ]
        )

        # Add visualization section
        jupyter_cells.extend(
            [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": "## Visualizations\n\nThis section contains key visualizations from our analysis.",
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": "### Dashboard Overview\n\n"
                    + f"The dashboard contains {len(dashboard.get('sections', []))} sections with various visualizations.",
                },
            ]
        )

        # Add recommendations section
        recommendations = []
        for dataset_name, finding in findings.items():
            if isinstance(finding, dict) and "recommendations" in finding:
                for rec in finding.get("recommendations", []):
                    recommendations.append(f"- **{dataset_name}**: {rec}")

        jupyter_cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "## Recommendations\n\nBased on our analysis, we recommend the following actions:\n\n"
                + "\n".join(recommendations),
            }
        )

        # Create summary
        report_summary = {
            "title": "Data Analysis Report",
            "date": "2023-04-13",
            "datasets_analyzed": list(summary.keys()),
            "key_findings": [],
        }

        # Add key findings from each dataset
        for dataset_name, finding in findings.items():
            if isinstance(finding, dict) and "summary" in finding:
                report_summary["key_findings"].append(
                    {
                        "dataset": dataset_name,
                        "summary": finding.get("summary", ""),
                        "key_variables": finding.get("key_variables", []),
                    }
                )

        # Create recommendations
        report_recommendations = {
            "data_quality": [
                "Implement data validation checks to prevent missing values",
                "Standardize data collection processes to ensure consistency",
            ],
            "analysis": [
                "Conduct further analysis on correlations between key variables",
                "Investigate seasonal patterns in time series data",
            ],
            "modeling": [
                "Deploy the best performing model in a production environment",
                "Regularly retrain models with new data to maintain accuracy",
                "Consider ensemble methods to improve prediction performance",
            ],
            "business_actions": [
                "Use insights to inform strategic decision making",
                "Develop a data-driven approach to problem solving",
                "Invest in data infrastructure to support ongoing analysis",
            ],
        }

        # Return the report
        return {
            "JupyterLogbook": {
                "cells": jupyter_cells,
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3",
                    },
                    "language_info": {
                        "codemirror_mode": {"name": "ipython", "version": 3},
                        "file_extension": ".py",
                        "mimetype": "text/x-python",
                        "name": "python",
                        "nbconvert_exporter": "python",
                        "pygments_lexer": "ipython3",
                        "version": "3.8.10",
                    },
                },
                "nbformat": 4,
                "nbformat_minor": 5,
            },
            "Report.summary": report_summary,
            "Report.recommendations": report_recommendations,
        }
