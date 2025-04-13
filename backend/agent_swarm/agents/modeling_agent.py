"""
Modeling Agent

This module defines the ModelingAgent class for developing and evaluating predictive models.
"""

from typing import Any, Dict, List, Optional
import logging

from .base_agent import BaseAgent
from ..llama_workflow.task_agents import ModelingTaskAgent


class ModelingAgent(BaseAgent):
    """
    Agent responsible for developing and evaluating predictive or descriptive models.
    
    The Modeling Agent:
    - Selects appropriate modeling algorithms
    - Performs feature engineering/selection
    - Manages data splitting, model training, and evaluation
    - Requests visualizations for model evaluation
    """
    
    def __init__(self):
        """Initialize the Modeling Agent."""
        super().__init__(name="ModelingAgent")
        self.logger = logging.getLogger(__name__)
        self.task_agent = ModelingTaskAgent()
    
    def run(self, environment: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute the agent's primary functionality.
        
        Args:
            environment: The shared environment state
            **kwargs: Additional arguments
                - data_reference: Reference to the data to model
                - target_variable: Name of the target variable
                - model_type: Optional type of model to build
                
        Returns:
            Dict containing:
                - models: Dict with model artifacts and metrics
                - visualization_requests: List of visualization requests
                - suggestions: List of suggested next steps
        """
        # Extract data reference from kwargs or environment
        data_reference = kwargs.get("data_reference")
        if not data_reference:
            if "cleaned_data" in environment:
                data_reference = environment["cleaned_data"]
            elif "loaded_data" in environment:
                data_reference = environment["loaded_data"]
        
        target_variable = kwargs.get("target_variable")
        model_type = kwargs.get("model_type")
        
        self.logger.info(f"Modeling data: {data_reference}, target: {target_variable}")
        
        # Use the task agent to build models
        task_input = {
            "environment": environment,
            "goals": ["Build predictive models"],
            "data_reference": data_reference,
            "target_variable": target_variable,
            "model_type": model_type
        }
        
        try:
            # Run the task agent
            result = self.task_agent.run(task_input)
            
            # Extract the results
            models = result.get("Models.trained_models", {})
            evaluation = result.get("Models.evaluation", {})
            
            # Create visualization requests based on models
            visualization_requests = [
                {"type": "roc_curve", "model": "model_1"},
                {"type": "confusion_matrix", "model": "model_1"}
            ]
            
            # Add suggestions for next steps
            suggestions = ["Run ReportingAgent to generate final report"]
            
            # Combine model information
            model_results = {}
            for model_name, model_info in models.items():
                model_results[model_name] = {
                    "type": model_info.get("type", "Unknown"),
                    "hyperparameters": model_info.get("hyperparameters", {}),
                    "performance": evaluation.get(model_name, {}),
                    "feature_importance": model_info.get("feature_importance", {})
                }
            
            return {
                "models": model_results,
                "visualization_requests": visualization_requests,
                "suggestions": suggestions
            }
        except Exception as e:
            self.logger.error(f"Error building models: {str(e)}")
            return {
                "error": str(e),
                "models": {},
                "visualization_requests": [],
                "suggestions": ["Check data format and try again"]
            }