"""
AI Agent Swarm for Data Science

This package implements an AI Agent Swarm for automating the end-to-end data science workflow.
The system leverages a multi-agent architecture built using LlamaIndex and powered by
Google's Gemini language model.
"""

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

__all__ = [
    "OrchestratorAgent",
    "DataLoadingAgent",
    "ExplorationAgent",
    "CleaningAgent",
    "AnalysisAgent",
    "ModelingAgent",
    "VisualizationAgent",
    "CodeActAgent",
    "ReportingAgent",
    "Environment"
]