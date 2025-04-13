"""
Custom Agent Workflow Module

This module provides custom implementations of the agent workflow components
for the agent framework.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging
from abc import ABC, abstractmethod
from datetime import datetime


class TaskAgent(ABC):
    """
    Abstract base class for task agents.
    
    Task agents are responsible for executing specific tasks within a workflow.
    """
    
    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's task.
        
        Args:
            input_data: Input data for the task
            
        Returns:
            Dict containing the task output
        """
        pass


class TaskOutput:
    """
    Container for task execution output.
    
    This class stores the output of a task execution along with metadata.
    """
    
    def __init__(self, output: Dict[str, Any], task_id: str):
        """
        Initialize a TaskOutput.
        
        Args:
            output: The output data from the task
            task_id: The ID of the task that produced this output
        """
        self.output = output
        self.task_id = task_id
        self.timestamp = datetime.now().isoformat()
    
    def __repr__(self) -> str:
        """Return a string representation of the TaskOutput."""
        return f"TaskOutput(task_id={self.task_id}, timestamp={self.timestamp})"


class Task:
    """
    Represents a task in a workflow.
    
    A task is a unit of work that can be executed by a task agent.
    """
    
    def __init__(
        self,
        task_id: str,
        agent: TaskAgent,
        description: str,
        input_data: Dict[str, Any],
        depends_on: Optional[List[str]] = None
    ):
        """
        Initialize a Task.
        
        Args:
            task_id: Unique identifier for the task
            agent: The agent that will execute this task
            description: Description of the task
            input_data: Input data for the task
            depends_on: List of task IDs that this task depends on
        """
        self.task_id = task_id
        self.agent = agent
        self.description = description
        self.input_data = input_data
        self.depends_on = depends_on or []
        self.output: Optional[TaskOutput] = None
        self.status = "pending"  # pending, running, completed, failed
        self.logger = logging.getLogger(__name__)
    
    def execute(self, dependency_outputs: Dict[str, TaskOutput] = None) -> TaskOutput:
        """
        Execute the task.
        
        Args:
            dependency_outputs: Outputs from dependency tasks
            
        Returns:
            TaskOutput containing the task output
        """
        dependency_outputs = dependency_outputs or {}
        
        # Update status
        self.status = "running"
        self.logger.info(f"Executing task: {self.task_id}")
        
        try:
            # Prepare input data with dependency outputs
            input_data = self.input_data.copy()
            
            # Add dependency outputs to input data
            for dep_id, dep_output in dependency_outputs.items():
                input_data[f"dependency_{dep_id}"] = dep_output.output
            
            # Execute the task
            output = self.agent.run(input_data)
            
            # Create task output
            self.output = TaskOutput(output=output, task_id=self.task_id)
            
            # Update status
            self.status = "completed"
            self.logger.info(f"Task completed: {self.task_id}")
            
            return self.output
        except Exception as e:
            # Update status
            self.status = "failed"
            self.logger.error(f"Task failed: {self.task_id} - {str(e)}")
            
            # Create error output
            error_output = {
                "error": str(e),
                "status": "failed"
            }
            self.output = TaskOutput(output=error_output, task_id=self.task_id)
            
            return self.output
    
    def __repr__(self) -> str:
        """Return a string representation of the Task."""
        return f"Task(id={self.task_id}, status={self.status}, depends_on={self.depends_on})"


class AgentWorkflow:
    """
    Manages the execution of a workflow of tasks.
    
    An AgentWorkflow is responsible for:
    - Tracking task dependencies
    - Executing tasks in the correct order
    - Managing task outputs
    """
    
    def __init__(self, tasks: List[Task]):
        """
        Initialize an AgentWorkflow.
        
        Args:
            tasks: List of tasks in the workflow
        """
        self.tasks = {task.task_id: task for task in tasks}
        self.task_outputs: Dict[str, TaskOutput] = {}
        self.logger = logging.getLogger(__name__)
        
        # Validate task dependencies
        self._validate_dependencies()
    
    def _validate_dependencies(self) -> None:
        """
        Validate that all task dependencies exist.
        
        Raises:
            ValueError: If a task depends on a non-existent task
        """
        for task_id, task in self.tasks.items():
            for dep_id in task.depends_on:
                if dep_id not in self.tasks:
                    raise ValueError(f"Task '{task_id}' depends on non-existent task '{dep_id}'")
    
    def _get_ready_tasks(self, completed_tasks: Set[str]) -> List[Task]:
        """
        Get tasks that are ready to be executed.
        
        A task is ready if all its dependencies have been completed.
        
        Args:
            completed_tasks: Set of completed task IDs
            
        Returns:
            List of tasks that are ready to be executed
        """
        ready_tasks = []
        
        for task_id, task in self.tasks.items():
            if task.status == "pending" and all(dep in completed_tasks for dep in task.depends_on):
                ready_tasks.append(task)
        
        return ready_tasks
    
    def execute(self) -> Dict[str, TaskOutput]:
        """
        Execute the workflow.
        
        Returns:
            Dict mapping task IDs to their outputs
        """
        self.logger.info("Starting workflow execution")
        
        completed_tasks: Set[str] = set()
        
        # Continue until all tasks are completed or no more tasks can be executed
        while len(completed_tasks) < len(self.tasks):
            # Get tasks that are ready to be executed
            ready_tasks = self._get_ready_tasks(completed_tasks)
            
            if not ready_tasks:
                # No more tasks can be executed, but not all tasks are completed
                # This could happen if there are circular dependencies
                self.logger.warning("No more tasks can be executed, but not all tasks are completed")
                break
            
            # Execute ready tasks
            for task in ready_tasks:
                # Get outputs from dependencies
                dependency_outputs = {
                    dep_id: self.task_outputs[dep_id]
                    for dep_id in task.depends_on
                }
                
                # Execute the task
                task_output = task.execute(dependency_outputs)
                
                # Store the output
                self.task_outputs[task.task_id] = task_output
                
                # Mark the task as completed
                completed_tasks.add(task.task_id)
        
        self.logger.info(f"Workflow execution completed. {len(completed_tasks)}/{len(self.tasks)} tasks completed.")
        
        return self.task_outputs