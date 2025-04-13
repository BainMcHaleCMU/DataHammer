"""
Cleaning Agent

This module defines the CleaningAgent class for preprocessing and cleaning data.
"""

from typing import Any, Dict, List, Optional
import logging

from .base_agent import BaseAgent
from ..llama_workflow.task_agents import CleaningTaskAgent


class CleaningAgent(BaseAgent):
    """
    Agent responsible for preprocessing and cleaning data.
    
    The Cleaning Agent:
    - Implements strategies for handling missing values
    - Handles outliers
    - Performs data type conversions
    - Addresses data sparsity issues
    - Documents cleaning steps applied
    """
    
    def __init__(self):
        """Initialize the Cleaning Agent."""
        super().__init__(name="CleaningAgent")
        self.logger = logging.getLogger(__name__)
        self.task_agent = CleaningTaskAgent()
    
    def run(self, environment: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute the agent's primary functionality.
        
        Args:
            environment: The shared environment state
            **kwargs: Additional arguments
                - data_reference: Reference to the data to clean
                - cleaning_strategies: Optional dict of cleaning strategies to apply
                
        Returns:
            Dict containing:
                - cleaned_data: Reference to cleaned data
                - cleaning_steps: List of cleaning steps applied
                - suggestions: List of suggested next steps
        """
        # Extract data reference from kwargs or environment
        data_reference = kwargs.get("data_reference")
        if not data_reference and "loaded_data" in environment:
            data_reference = environment["loaded_data"]
        
        cleaning_strategies = kwargs.get("cleaning_strategies", {})
        
        self.logger.info(f"Cleaning data: {data_reference}")
        
        # Use the task agent to clean the data
        task_input = {
            "environment": environment,
            "goals": ["Clean and preprocess data for analysis"],
            "data_reference": data_reference,
            "cleaning_strategies": cleaning_strategies
        }
        
        try:
            # Run the task agent
            result = self.task_agent.run(task_input)
            
            # Extract the results
            processed_data = result.get("Cleaned Data.processed_data", {})
            cleaning_steps = result.get("Cleaned Data.cleaning_steps", [])
            
            # Add suggestions for next steps
            suggestions = ["Run ExplorationAgent again on cleaned data"]
            
            return {
                "cleaned_data": processed_data,
                "cleaning_steps": cleaning_steps,
                "suggestions": suggestions
            }
        except Exception as e:
            self.logger.error(f"Error cleaning data: {str(e)}")
            return {
                "error": str(e),
                "cleaned_data": None,
                "cleaning_steps": [],
                "suggestions": ["Check data format and try again"]
            }