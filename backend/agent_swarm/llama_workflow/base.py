"""
Base Task Agent Module

This module defines the base task agent class used in agent workflows.
"""

from typing import Any, Dict, List, Optional
import logging
from abc import ABC, abstractmethod

from ..custom_framework.llm import LLM
from ..custom_framework.settings import Settings
from ..custom_framework.agent_workflow import TaskAgent


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
