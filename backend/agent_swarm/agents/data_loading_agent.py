"""
Data Loading Agent

This module defines the DataLoadingAgent class for ingesting data from various sources.
"""

from typing import Any, Dict, List, Optional
import logging
import pandas as pd
import json
import os

from .base_agent import BaseAgent
from ..llama_workflow.task_agents import DataLoadingTaskAgent


class DataLoadingAgent(BaseAgent):
    """
    Agent responsible for ingesting data from various sources.
    
    The Data Loading Agent:
    - Parses files (PDF, CSV, etc.)
    - Connects to databases
    - Loads data into standard formats (e.g., Pandas DataFrame)
    - Handles initial loading errors
    - Reports loaded data location/reference and initial schema
    """
    
    def __init__(self):
        """Initialize the Data Loading Agent."""
        super().__init__(name="DataLoadingAgent")
        self.logger = logging.getLogger(__name__)
        self.task_agent = DataLoadingTaskAgent()
    
    def run(self, environment: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute the agent's primary functionality.
        
        Args:
            environment: The shared environment state
            **kwargs: Additional arguments
                - data_sources: Dict of data source references
                - file_type: Optional type of file to load
                
        Returns:
            Dict containing:
                - loaded_data: Reference to loaded data
                - schema: Initial data schema
                - suggestions: List of suggested next steps
        """
        # Extract data sources from kwargs or environment
        data_sources = kwargs.get("data_sources", {})
        if not data_sources and "data_sources" in environment:
            data_sources = environment["data_sources"]
        
        file_type = kwargs.get("file_type")
        
        self.logger.info(f"Loading data from {len(data_sources)} sources")
        
        # Use the task agent to load the data
        task_input = {
            "environment": environment,
            "goals": ["Load and prepare data for analysis"],
            "data_sources": data_sources,
            "file_type": file_type
        }
        
        try:
            # Run the task agent
            result = self.task_agent.run(task_input)
            
            # Extract the results
            loaded_data = result.get("Data Overview.raw_data", {})
            schema = result.get("Data Overview.schema", {})
            
            # Add suggestions for next steps
            suggestions = ["Run ExplorationAgent to analyze the loaded data"]
            
            return {
                "loaded_data": loaded_data,
                "schema": schema,
                "suggestions": suggestions
            }
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return {
                "error": str(e),
                "loaded_data": None,
                "schema": None,
                "suggestions": ["Check data source paths and formats"]
            }