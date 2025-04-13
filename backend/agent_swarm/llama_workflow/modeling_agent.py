"""
Modeling Task Agent Module

This module defines the modeling task agent used in LlamaIndex agent workflows.
"""

from typing import Any, Dict, Optional
import logging

from llama_index.core.llms import LLM

from .base import BaseTaskAgent


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
                    self.logger.warning(
                        f"Skipping dataset {dataset_name} due to processing error"
                    )
                    continue

                # Get dataset findings
                dataset_findings = findings.get(dataset_name, {})

                # Determine model type based on goals and data
                # For simplicity, we'll simulate different model types
                model_types = [
                    "linear_regression",
                    "random_forest",
                    "gradient_boosting",
                ]

                # Train multiple models for each dataset
                trained_models[dataset_name] = {}
                performance_metrics[dataset_name] = {}

                for model_type in model_types:
                    # Simulate model training
                    self.logger.info(f"Training {model_type} model for {dataset_name}")

                    # Create model metadata
                    trained_models[dataset_name][model_type] = {
                        "type": model_type,
                        "features": dataset_findings.get("key_variables", [])
                        + ["additional_feature"],
                        "target": "target_variable",  # Simulated target variable
                        "hyperparameters": self._get_default_hyperparameters(
                            model_type
                        ),
                        "training_time": "2.5 seconds",  # Simulated training time
                        "timestamp": "2023-04-13T10:30:00Z",  # Simulated timestamp
                    }

                    # Create performance metrics
                    performance_metrics[dataset_name][model_type] = (
                        self._generate_performance_metrics(model_type)
                    )

                # Determine best model
                best_model = max(
                    performance_metrics[dataset_name].items(),
                    key=lambda x: (
                        x[1].get("metrics", {}).get("r2", 0)
                        if "regression" in x[0]
                        else x[1].get("metrics", {}).get("accuracy", 0)
                    ),
                )[0]

                performance_metrics[dataset_name]["best_model"] = best_model

            except Exception as e:
                self.logger.error(f"Error modeling dataset {dataset_name}: {str(e)}")
                trained_models[dataset_name] = {"error": str(e)}
                performance_metrics[dataset_name] = {"error": str(e)}

        # Return the modeling results
        return {
            "Models.trained_model": trained_models,
            "Models.performance": performance_metrics,
        }

    def _get_default_hyperparameters(self, model_type: str) -> Dict[str, Any]:
        """Get default hyperparameters for a model type."""
        if model_type == "linear_regression":
            return {"fit_intercept": True, "normalize": False}
        elif model_type == "random_forest":
            return {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
            }
        elif model_type == "gradient_boosting":
            return {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "subsample": 0.8,
            }
        else:
            return {}

    def _generate_performance_metrics(self, model_type: str) -> Dict[str, Any]:
        """Generate simulated performance metrics for a model."""
        if "regression" in model_type:
            return {
                "metrics": {
                    "r2": (
                        0.85
                        if model_type == "gradient_boosting"
                        else (0.82 if model_type == "random_forest" else 0.75)
                    ),
                    "mse": (
                        0.15
                        if model_type == "gradient_boosting"
                        else (0.18 if model_type == "random_forest" else 0.25)
                    ),
                    "mae": (
                        0.12
                        if model_type == "gradient_boosting"
                        else (0.15 if model_type == "random_forest" else 0.20)
                    ),
                },
                "cross_validation": {
                    "method": "5-fold",
                    "r2_scores": (
                        [0.83, 0.86, 0.84, 0.87, 0.85]
                        if model_type == "gradient_boosting"
                        else (
                            [0.80, 0.83, 0.81, 0.84, 0.82]
                            if model_type == "random_forest"
                            else [0.73, 0.76, 0.74, 0.77, 0.75]
                        )
                    ),
                },
            }
        else:
            return {
                "metrics": {
                    "accuracy": (
                        0.92
                        if model_type == "gradient_boosting"
                        else (0.90 if model_type == "random_forest" else 0.85)
                    ),
                    "precision": (
                        0.91
                        if model_type == "gradient_boosting"
                        else (0.89 if model_type == "random_forest" else 0.84)
                    ),
                    "recall": (
                        0.90
                        if model_type == "gradient_boosting"
                        else (0.88 if model_type == "random_forest" else 0.83)
                    ),
                    "f1": (
                        0.905
                        if model_type == "gradient_boosting"
                        else (0.885 if model_type == "random_forest" else 0.835)
                    ),
                },
                "cross_validation": {
                    "method": "5-fold",
                    "accuracy_scores": (
                        [0.91, 0.93, 0.92, 0.94, 0.92]
                        if model_type == "gradient_boosting"
                        else (
                            [0.89, 0.91, 0.90, 0.92, 0.90]
                            if model_type == "random_forest"
                            else [0.84, 0.86, 0.85, 0.87, 0.85]
                        )
                    ),
                },
            }
