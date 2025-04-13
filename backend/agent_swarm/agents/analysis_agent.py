"""
Analysis Agent

This module defines the AnalysisAgent class for deriving deeper insights from data.
"""

from typing import Any, Dict, List, Optional
import logging

from .base_agent import BaseAgent
from ..llama_workflow.task_agents import AnalysisTaskAgent


class AnalysisAgent(BaseAgent):
    """
    Agent responsible for deriving deeper insights from data relevant to the goals.
    
    The Analysis Agent:
    - Performs targeted statistical analyses
    - Conducts hypothesis testing
    - Investigates complex correlations
    - Analyzes outliers for insights
    - Performs segmentation if needed
    """
    
    def __init__(self):
        """Initialize the Analysis Agent."""
        super().__init__(name="AnalysisAgent")
        self.logger = logging.getLogger(__name__)
        self.task_agent = AnalysisTaskAgent()
    
    def run(self, environment: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute the agent's primary functionality.
        
        Args:
            environment: The shared environment state
            **kwargs: Additional arguments
                - data_reference: Reference to the data to analyze
                - analysis_targets: Optional list of specific analysis targets
                
        Returns:
            Dict containing:
                - analysis_results: Dict with analysis findings
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
        
        analysis_targets = kwargs.get("analysis_targets", [])
        
        self.logger.info(f"Analyzing data: {data_reference}")
        
        # Use the task agent to analyze the data
        task_input = {
            "environment": environment,
            "goals": ["Derive deeper insights from data"],
            "data_reference": data_reference,
            "analysis_targets": analysis_targets
        }
        
        try:
            # Run the task agent
            result = self.task_agent.run(task_input)
            
            # Extract the results
            analysis_results = result.get("Analysis.findings", {})
            insights = result.get("Analysis.insights", [])
            
            # Create visualization requests based on findings
            visualization_requests = [
                {"type": "scatter", "x": "feature_x", "y": "feature_y"},
                {"type": "box_plot", "data": "feature_z", "by": "segment"}
            ]
            
            # Add suggestions for next steps
            suggestions = ["Run ModelingAgent to predict target variable"]
            
            return {
                "analysis_results": {
                    "statistical_tests": analysis_results.get("statistical_tests", {}),
                    "correlations": analysis_results.get("correlations", {}),
                    "segments": analysis_results.get("segments", {})
                },
                "insights": insights,
                "visualization_requests": visualization_requests,
                "suggestions": suggestions
            }
        except Exception as e:
            self.logger.error(f"Error analyzing data: {str(e)}")
            return {
                "error": str(e),
                "analysis_results": {
                    "statistical_tests": {},
                    "correlations": {},
                    "segments": {}
                },
                "visualization_requests": [],
                "suggestions": ["Check data format and try again"]
            }