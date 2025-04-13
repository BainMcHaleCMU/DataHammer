"""
Reporting Agent

This module defines the ReportingAgent class for compiling final results into a report.
"""

from typing import Any, Dict, List, Optional
import logging

from .base_agent import BaseAgent
from ..llama_workflow.task_agents import ReportingTaskAgent


class ReportingAgent(BaseAgent):
    """
    Agent responsible for compiling final results and process into a coherent report.
    
    The Reporting Agent:
    - Gathers key information from across the Environment
    - Synthesizes narrative explanations and summaries
    - Requests final summary visualizations
    - Produces the final output (polished JupyterLogbook or Markdown)
    """
    
    def __init__(self):
        """Initialize the Reporting Agent."""
        super().__init__(name="ReportingAgent")
        self.logger = logging.getLogger(__name__)
        self.task_agent = ReportingTaskAgent()
    
    def run(self, environment: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute the agent's primary functionality.
        
        Args:
            environment: The shared environment state
            **kwargs: Additional arguments
                - report_format: Optional format for the report
                - sections: Optional list of sections to include
                
        Returns:
            Dict containing:
                - report: The final report content or reference
                - visualization_requests: List of final visualization requests
        """
        # Extract report format and sections from kwargs
        report_format = kwargs.get("report_format", "jupyter")
        sections = kwargs.get("sections", [])
        
        self.logger.info(f"Generating report in {report_format} format")
        
        # Use the task agent to generate the report
        task_input = {
            "environment": environment,
            "goals": ["Generate comprehensive data analysis report"],
            "report_format": report_format,
            "sections": sections
        }
        
        try:
            # Run the task agent
            result = self.task_agent.run(task_input)
            
            # Extract the results
            report_path = result.get("Report.path", "path/to/final_report.ipynb")
            report_content = result.get("Report.content", {})
            
            # Generate final visualization requests
            visualization_requests = [
                {"type": "summary_plot", "data": "all_results"}
            ]
            
            return {
                "report": report_path,
                "report_content": report_content,
                "visualization_requests": visualization_requests
            }
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return {
                "error": str(e),
                "report": None,
                "visualization_requests": []
            }