"""
Orchestrator Agent

This module defines the OrchestratorAgent class that coordinates the AI Agent Swarm.
"""

from typing import Any, Dict, List, Optional, Type
import logging
from datetime import datetime

from .base_agent import BaseAgent
from ..workflow import WorkflowPlanner, WorkflowExecutor, WorkflowGraph


class OrchestratorAgent(BaseAgent):
    """
    Central coordinator for the AI Agent Swarm.
    
    The Orchestrator Agent is responsible for:
    - Initializing the Environment
    - Dynamically planning the workflow
    - Invoking specialized agents
    - Managing the Environment state
    - Updating the JupyterLogbook
    - Handling errors and coordinating corrective actions
    """
    
    def __init__(self):
        """Initialize the Orchestrator Agent."""
        super().__init__(name="OrchestratorAgent")
        self.available_agents = {}
        self.environment = self._initialize_environment()
        self.workflow_planner = None
        self.workflow_executor = None
        self.logger = logging.getLogger(__name__)
    
    def _initialize_environment(self) -> Dict[str, Any]:
        """
        Initialize the shared Environment state.
        
        Returns:
            Dict containing the initial Environment state
        """
        return {
            "Goals": [],
            "Data": {},
            "Data Overview": {},
            "Cleaned Data": {},
            "Analysis Results": {},
            "Models": {},
            "Visualizations": {},
            "JupyterLogbook": None,
            "Available Agents": {},
            "Execution State/Log": [],
            "Workflow": {
                "Current": None,
                "History": []
            }
        }
    
    def register_agent(self, agent_class: Type[BaseAgent], agent_name: Optional[str] = None) -> None:
        """
        Register a specialized agent with the Orchestrator.
        
        Args:
            agent_class: The agent class to register
            agent_name: Optional custom name for the agent
        """
        agent_instance = agent_class()
        name = agent_name or agent_instance.name
        self.available_agents[name] = agent_instance
        
        # Update the Available Agents in the environment
        self.environment["Available Agents"][name] = {
            "name": name,
            "description": agent_instance.__class__.__doc__
        }
    
    def run(self, environment: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute the Orchestrator's primary functionality.
        
        This method implements the dynamic workflow planning and execution.
        
        Args:
            environment: Optional external environment state to use
            **kwargs: Additional arguments
                - goals: List of user-defined goals
                - data_sources: Dict of data source references
                
        Returns:
            Dict containing the final Environment state
        """
        # Use provided environment or the internal one
        env = environment if environment is not None else self.environment
        
        # Update goals and data if provided
        if "goals" in kwargs:
            env["Goals"] = kwargs["goals"]
        
        if "data_sources" in kwargs:
            env["Data"] = kwargs["data_sources"]
        
        # Initialize JupyterLogbook if not already done
        if env["JupyterLogbook"] is None:
            env["JupyterLogbook"] = self._initialize_jupyter_logbook()
        
        # Log the initialization
        self._log_execution_state("Orchestrator initialized", "System initialized with goals and data sources")
        
        # Initialize workflow components if not already done
        if self.workflow_planner is None:
            self.workflow_planner = WorkflowPlanner()
        
        if self.workflow_executor is None:
            self.workflow_executor = WorkflowExecutor()
        
        # Plan the workflow
        self._log_execution_state("Planning workflow", f"Planning workflow based on goals: {env['Goals']}")
        workflow = self.workflow_planner.create_workflow(env["Goals"], env)
        
        # Store the workflow in the environment
        env["Workflow"]["Current"] = {
            "steps": {step.id: {
                "id": step.id,
                "agent_name": step.agent_name,
                "task": step.task,
                "description": step.description,
                "status": step.status,
                "dependencies": step.dependencies
            } for step in workflow.get_all_steps()},
            "execution_order": workflow.get_execution_order()
        }
        
        # Log the workflow plan
        self._log_execution_state(
            "Workflow planned", 
            f"Created workflow with {len(workflow.get_all_steps())} steps"
        )
        
        # Execute the workflow
        self._log_execution_state("Executing workflow", "Starting workflow execution")
        updated_env = self.workflow_executor.execute_workflow(
            workflow, env, self.invoke_agent
        )
        
        # Update the environment with the workflow execution results
        env.update(updated_env)
        
        # Update workflow status in the environment
        env["Workflow"]["Current"]["status"] = {
            "completed_steps": list(workflow.completed_steps),
            "failed_steps": list(workflow.failed_steps),
            "total_steps": len(workflow.steps)
        }
        
        # Archive the current workflow in history
        env["Workflow"]["History"].append(env["Workflow"]["Current"])
        
        # Log the completion
        self._log_execution_state(
            "Workflow execution completed", 
            f"Completed {len(workflow.completed_steps)}/{len(workflow.steps)} steps"
        )
        
        return env
    
    def _initialize_jupyter_logbook(self) -> Any:
        """
        Initialize the JupyterLogbook.
        
        Returns:
            An initialized notebook object
        """
        # TODO: Implement notebook initialization using nbformat
        return {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    
    def _log_execution_state(self, action: str, details: str) -> None:
        """
        Log an action in the Execution State/Log.
        
        Args:
            action: The action being logged
            details: Details about the action
        """
        log_entry = {
            "action": action,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.environment["Execution State/Log"].append(log_entry)
        self.logger.info(f"{action}: {details}")
    
    def invoke_agent(self, agent_name: str, **kwargs) -> Dict[str, Any]:
        """
        Invoke a specialized agent.
        
        Args:
            agent_name: Name of the agent to invoke
            **kwargs: Additional arguments to pass to the agent
            
        Returns:
            Dict containing the agent's results
            
        Raises:
            ValueError: If the agent is not registered
        """
        if agent_name not in self.available_agents:
            raise ValueError(f"Agent '{agent_name}' is not registered")
        
        agent = self.available_agents[agent_name]
        
        # Log the agent invocation
        self._log_execution_state(
            f"Invoking {agent_name}",
            f"Delegating task to {agent_name} with args: {kwargs}"
        )
        
        # Run the agent with the current environment
        result = agent.run(self.environment, **kwargs)
        
        # Log the agent completion
        self._log_execution_state(
            f"{agent_name} completed",
            f"Agent completed task: {kwargs.get('task', 'unknown')}"
        )
        
        return result
    
    def update_jupyter_logbook(self, markdown_content: str = None, code_content: str = None) -> None:
        """
        Update the JupyterLogbook with new content.
        
        Args:
            markdown_content: Optional markdown content to add
            code_content: Optional code content to add
        """
        # TODO: Implement notebook update logic using nbformat
        pass
    
    def replan_workflow(self, environment: Dict[str, Any] = None) -> WorkflowGraph:
        """
        Replan the workflow based on the current environment state.
        
        This is useful when the environment has changed significantly or when
        the current workflow has failed steps.
        
        Args:
            environment: Optional external environment state to use
            
        Returns:
            The new workflow graph
        """
        # Use provided environment or the internal one
        env = environment if environment is not None else self.environment
        
        # Log the replanning
        self._log_execution_state("Replanning workflow", "Replanning workflow based on current state")
        
        # Create a new workflow plan
        workflow = self.workflow_planner.create_workflow(env["Goals"], env)
        
        # Store the workflow in the environment
        env["Workflow"]["Current"] = {
            "steps": {step.id: {
                "id": step.id,
                "agent_name": step.agent_name,
                "task": step.task,
                "description": step.description,
                "status": step.status,
                "dependencies": step.dependencies
            } for step in workflow.get_all_steps()},
            "execution_order": workflow.get_execution_order()
        }
        
        # Log the workflow plan
        self._log_execution_state(
            "Workflow replanned", 
            f"Created new workflow with {len(workflow.get_all_steps())} steps"
        )
        
        return workflow