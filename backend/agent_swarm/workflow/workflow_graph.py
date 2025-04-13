"""
Workflow Graph Module

This module defines the WorkflowGraph class that represents a directed graph of workflow steps.
"""

from typing import Any, Dict, List, Optional, Set
import networkx as nx
from .workflow_step import WorkflowStep


class WorkflowGraph:
    """
    Represents a directed graph of workflow steps.
    
    The WorkflowGraph is responsible for:
    - Storing the workflow steps and their dependencies
    - Validating the workflow structure
    - Identifying the next steps to execute
    - Tracking the execution state
    """
    
    def __init__(self):
        """Initialize an empty workflow graph."""
        self.graph = nx.DiGraph()
        self.steps: Dict[str, WorkflowStep] = {}
        self.completed_steps: Set[str] = set()
        self.failed_steps: Set[str] = set()
        self.in_progress_steps: Set[str] = set()
    
    def add_step(self, step: WorkflowStep) -> None:
        """
        Add a step to the workflow graph.
        
        Args:
            step: The workflow step to add
        """
        if step.id in self.steps:
            raise ValueError(f"Step with ID '{step.id}' already exists")
        
        self.steps[step.id] = step
        self.graph.add_node(step.id, step=step)
        
        # Add edges for dependencies
        for dep_id in step.dependencies:
            if dep_id not in self.steps:
                raise ValueError(f"Dependency '{dep_id}' not found for step '{step.id}'")
            self.graph.add_edge(dep_id, step.id)
    
    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        """
        Get a step by its ID.
        
        Args:
            step_id: The ID of the step to get
            
        Returns:
            The workflow step, or None if not found
        """
        return self.steps.get(step_id)
    
    def get_next_steps(self) -> List[WorkflowStep]:
        """
        Get the next steps that are ready to be executed.
        
        A step is ready when all its dependencies have been completed.
        
        Returns:
            List of workflow steps that are ready to be executed
        """
        ready_steps = []
        for step_id, step in self.steps.items():
            if (step_id not in self.completed_steps and 
                step_id not in self.failed_steps and
                step_id not in self.in_progress_steps and
                step.is_ready(list(self.completed_steps))):
                ready_steps.append(step)
        
        return ready_steps
    
    def mark_step_completed(self, step_id: str, result: Dict[str, Any] = None) -> None:
        """
        Mark a step as completed.
        
        Args:
            step_id: The ID of the step to mark as completed
            result: Optional result of the step execution
        """
        if step_id not in self.steps:
            raise ValueError(f"Step '{step_id}' not found")
        
        step = self.steps[step_id]
        step.status = "completed"
        step.result = result
        
        self.completed_steps.add(step_id)
        self.in_progress_steps.discard(step_id)
    
    def mark_step_failed(self, step_id: str, error: str = None) -> None:
        """
        Mark a step as failed.
        
        Args:
            step_id: The ID of the step to mark as failed
            error: Optional error message
        """
        if step_id not in self.steps:
            raise ValueError(f"Step '{step_id}' not found")
        
        step = self.steps[step_id]
        step.status = "failed"
        if error:
            step.result = {"error": error}
        
        self.failed_steps.add(step_id)
        self.in_progress_steps.discard(step_id)
    
    def mark_step_in_progress(self, step_id: str) -> None:
        """
        Mark a step as in progress.
        
        Args:
            step_id: The ID of the step to mark as in progress
        """
        if step_id not in self.steps:
            raise ValueError(f"Step '{step_id}' not found")
        
        step = self.steps[step_id]
        step.status = "in_progress"
        
        self.in_progress_steps.add(step_id)
    
    def is_workflow_completed(self) -> bool:
        """
        Check if the workflow is completed.
        
        A workflow is completed when all steps are either completed or failed.
        
        Returns:
            True if the workflow is completed, False otherwise
        """
        return len(self.completed_steps) + len(self.failed_steps) == len(self.steps)
    
    def validate(self) -> bool:
        """
        Validate the workflow graph.
        
        Checks for cycles and missing dependencies.
        
        Returns:
            True if the workflow is valid, False otherwise
            
        Raises:
            ValueError: If the workflow contains cycles
        """
        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            cycles = list(nx.simple_cycles(self.graph))
            raise ValueError(f"Workflow contains cycles: {cycles}")
        
        # Check for missing dependencies
        for step_id, step in self.steps.items():
            for dep_id in step.dependencies:
                if dep_id not in self.steps:
                    raise ValueError(f"Dependency '{dep_id}' not found for step '{step_id}'")
        
        return True
    
    def get_execution_order(self) -> List[str]:
        """
        Get the topological order of steps.
        
        Returns:
            List of step IDs in topological order
        """
        return list(nx.topological_sort(self.graph))
    
    def get_all_steps(self) -> List[WorkflowStep]:
        """
        Get all steps in the workflow.
        
        Returns:
            List of all workflow steps
        """
        return list(self.steps.values())
    
    def get_step_status(self) -> Dict[str, str]:
        """
        Get the status of all steps.
        
        Returns:
            Dict mapping step IDs to their status
        """
        return {step_id: step.status for step_id, step in self.steps.items()}