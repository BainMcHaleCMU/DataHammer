"""
Workflow Step Module

This module defines the WorkflowStep class that represents a single step in a workflow.
"""

from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field


@dataclass
class WorkflowStep:
    """
    Represents a single step in a workflow.
    
    A WorkflowStep encapsulates:
    - The agent responsible for executing the step
    - The task to be performed
    - Input requirements
    - Output specifications
    - Dependencies on other steps
    """
    
    id: str
    agent_name: str
    task: str
    description: str
    input_keys: List[str] = field(default_factory=list)
    output_keys: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[Dict[str, Any]] = None
    
    def is_ready(self, completed_steps: List[str]) -> bool:
        """
        Check if this step is ready to be executed.
        
        A step is ready when all its dependencies have been completed.
        
        Args:
            completed_steps: List of IDs of completed workflow steps
            
        Returns:
            True if the step is ready to be executed, False otherwise
        """
        return all(dep in completed_steps for dep in self.dependencies)
    
    def get_input_from_environment(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract the required inputs for this step from the environment.
        
        Args:
            environment: The shared environment state
            
        Returns:
            Dict containing the inputs for this step
        """
        inputs = {}
        for key in self.input_keys:
            # Handle nested keys with dot notation (e.g., "Data.csv_file")
            if "." in key:
                parts = key.split(".")
                value = environment
                for part in parts:
                    if part in value:
                        value = value[part]
                    else:
                        value = None
                        break
                inputs[key] = value
            else:
                inputs[key] = environment.get(key)
        
        # Add any additional parameters
        inputs.update(self.parameters)
        
        return inputs
    
    def update_environment_with_output(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the environment with the output of this step.
        
        Args:
            environment: The shared environment state
            
        Returns:
            Updated environment
        """
        if not self.result:
            return environment
        
        # Create a copy of the environment to avoid modifying the original
        updated_env = environment.copy()
        
        for key in self.output_keys:
            if key in self.result:
                # Handle nested keys with dot notation
                if "." in key:
                    parts = key.split(".")
                    target = updated_env
                    for part in parts[:-1]:
                        if part not in target:
                            target[part] = {}
                        target = target[part]
                    target[parts[-1]] = self.result[key]
                else:
                    updated_env[key] = self.result[key]
        
        return updated_env