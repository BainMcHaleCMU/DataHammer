"""
Base Task Agent Module

This module defines the base task agent class used in LlamaIndex agent workflows.
"""

from typing import Any, Dict, List, Optional
import logging
from abc import ABC, abstractmethod

# Define a simple TaskAgent class since we can't import it from llama_index
class TaskAgent:
    """Base class for task agents in the workflow."""
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's primary functionality."""
        raise NotImplementedError("Subclasses must implement run method")

# Define a simple LLM class
class LLM:
    """Base class for language models."""
    pass

# Define a simple Settings class
class Settings:
    """Settings class for LlamaIndex."""
    llm = None


class BaseTaskAgent(TaskAgent, ABC):
    """
    Base class for all task agents in the workflow.

    All specialized task agents inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self, name: str, llm: Optional[LLM] = None):
        """
        Initialize the base task agent.

        Args:
            name: The name of the agent
            llm: Optional language model to use
        """
        self.name = name
        self.llm = llm or Settings.llm
        if not self.llm:
            raise ValueError(
                "No language model available. Please configure Settings.llm or provide an LLM."
            )

        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's primary functionality.

        Args:
            input_data: Input data for the task

        Returns:
            Dict containing results and any suggestions for next steps
        """
        pass
