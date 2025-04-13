"""
Visualization Agent

This module defines the VisualizationAgent class for generating visual representations.
"""

from typing import Any, Dict, List, Optional
import logging

from .base_agent import BaseAgent
from ..llama_workflow.task_agents import VisualizationTaskAgent


class VisualizationAgent(BaseAgent):
    """
    Agent responsible for generating visual representations of data and results.
    
    The Visualization Agent:
    - Receives requests for specific plot types
    - Formulates Python code using plotting libraries
    - Uses CodeActAgent to execute plotting code
    - Returns plot references and descriptions
    """
    
    def __init__(self):
        """Initialize the Visualization Agent."""
        super().__init__(name="VisualizationAgent")
        self.logger = logging.getLogger(__name__)
        self.task_agent = VisualizationTaskAgent()
    
    def run(self, environment: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute the agent's primary functionality.
        
        Args:
            environment: The shared environment state
            **kwargs: Additional arguments
                - data_reference: Reference to the data to visualize
                - plot_type: Type of plot to generate
                - plot_params: Dict of parameters for the plot
                
        Returns:
            Dict containing:
                - plot_reference: Path to the generated plot
                - plot_description: Description of the plot
        """
        # Extract data reference from kwargs or environment
        data_reference = kwargs.get("data_reference")
        if not data_reference:
            if "cleaned_data" in environment:
                data_reference = environment["cleaned_data"]
            elif "loaded_data" in environment:
                data_reference = environment["loaded_data"]
        
        plot_type = kwargs.get("plot_type", "histogram")
        plot_params = kwargs.get("plot_params", {})
        
        self.logger.info(f"Creating visualization: {plot_type} for {data_reference}")
        
        # Use the task agent to create visualizations
        task_input = {
            "environment": environment,
            "goals": ["Create informative visualizations"],
            "data_reference": data_reference,
            "plot_type": plot_type,
            "plot_params": plot_params
        }
        
        try:
            # Run the task agent
            result = self.task_agent.run(task_input)
            
            # Extract the results
            plot_reference = result.get("Visualization.plot_path", "path/to/plot.png")
            plot_description = result.get("Visualization.description", f"{plot_type} visualization")
            
            return {
                "plot_reference": plot_reference,
                "plot_description": plot_description
            }
        except Exception as e:
            self.logger.error(f"Error creating visualization: {str(e)}")
            return {
                "error": str(e),
                "plot_reference": None,
                "plot_description": f"Failed to create {plot_type} visualization"
            }