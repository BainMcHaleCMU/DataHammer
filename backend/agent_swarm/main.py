"""
AI Agent Swarm for Data Science - Main Module

This module provides the main entry point for the AI Agent Swarm system.
"""

from typing import Any, Dict, List, Optional
import os
import argparse
import json
import logging

from .custom_framework.llm import Gemini
from .custom_framework.settings import Settings

from .agents import (
    OrchestratorAgent,
    DataLoadingAgent,
    ExplorationAgent,
    CleaningAgent,
    AnalysisAgent,
    ModelingAgent,
    VisualizationAgent,
    CodeActAgent,
    ReportingAgent
)
from .environment import Environment
from .llama_workflow import WorkflowManager


def setup_llm(api_key: Optional[str] = None):
    """
    Set up the language model for the agent swarm.
    
    Args:
        api_key: Optional Google API key
    """
    # Use provided API key or get from environment
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    # Verify the correct model identifier string with Google docs
    flash_model_name = "models/gemini-2.0-flash-lite"
    Settings.llm = Gemini(model=flash_model_name)
    
    # Configure embedding model if needed
    # Settings.embed_model = ...


def create_agent_swarm() -> OrchestratorAgent:
    """
    Create and configure the AI Agent Swarm.
    
    Returns:
        The configured Orchestrator Agent
    """
    # Create the Orchestrator Agent
    orchestrator = OrchestratorAgent()
    
    # Register specialized agents
    orchestrator.register_agent(DataLoadingAgent)
    orchestrator.register_agent(ExplorationAgent)
    orchestrator.register_agent(CleaningAgent)
    orchestrator.register_agent(AnalysisAgent)
    orchestrator.register_agent(ModelingAgent)
    orchestrator.register_agent(VisualizationAgent)
    orchestrator.register_agent(CodeActAgent)
    orchestrator.register_agent(ReportingAgent)
    
    return orchestrator


def run_agent_swarm(goals: List[str], data_sources: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the AI Agent Swarm with the specified goals and data sources.
    
    Args:
        goals: List of user-defined goals
        data_sources: Dict of data source references
        
    Returns:
        Dict containing the final Environment state
    """
    # Set up the language model
    setup_llm()
    
    # Create the agent swarm
    orchestrator = create_agent_swarm()
    
    # Run the Orchestrator with the provided goals and data sources
    result = orchestrator.run(goals=goals, data_sources=data_sources)
    
    return result


def main():
    """Main entry point for the AI Agent Swarm."""
    parser = argparse.ArgumentParser(description="AI Agent Swarm for Data Science")
    parser.add_argument("--goals", type=str, help="JSON file containing goals")
    parser.add_argument("--data", type=str, help="JSON file containing data source references")
    parser.add_argument("--api-key", type=str, help="Google API key")
    args = parser.parse_args()
    
    # Load goals
    if args.goals:
        with open(args.goals, 'r') as f:
            goals = json.load(f)
    else:
        goals = ["Explore the data", "Identify key insights", "Build a predictive model"]
    
    # Load data sources
    if args.data:
        with open(args.data, 'r') as f:
            data_sources = json.load(f)
    else:
        data_sources = {"csv_file": "path/to/data.csv"}
    
    # Set up the language model
    setup_llm(args.api_key)
    
    # Run the agent swarm
    result = run_agent_swarm(goals, data_sources)
    
    print("Agent Swarm execution completed.")
    print(f"Results saved to: {result.get('JupyterLogbook', 'No logbook generated')}")


if __name__ == "__main__":
    main()