"""
Workflow Executor Module

This module defines the WorkflowExecutor class that executes workflow steps.
"""

from typing import Any, Dict, List, Optional, Callable
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from .workflow_step import WorkflowStep
from .workflow_graph import WorkflowGraph


class WorkflowExecutor:
    """
    Executes workflow steps using the appropriate agents.
    
    The WorkflowExecutor is responsible for:
    - Executing workflow steps in the correct order
    - Managing parallel execution when possible
    - Handling step failures and retries
    - Updating the environment with step results
    """
    
    def __init__(self, max_parallel: int = 3, max_retries: int = 2):
        """
        Initialize the WorkflowExecutor.
        
        Args:
            max_parallel: Maximum number of steps to execute in parallel
            max_retries: Maximum number of retries for failed steps
        """
        self.max_parallel = max_parallel
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)
        self.retry_counts: Dict[str, int] = {}
    
    def execute_workflow(self, workflow: WorkflowGraph, environment: Dict[str, Any], 
                        invoke_agent_fn: Callable) -> Dict[str, Any]:
        """
        Execute a workflow using the provided agent invocation function.
        
        Args:
            workflow: The workflow graph to execute
            environment: The shared environment state
            invoke_agent_fn: Function to invoke an agent (signature: invoke_agent_fn(agent_name, **kwargs))
            
        Returns:
            Updated environment after workflow execution
        """
        # Create a copy of the environment to avoid modifying the original
        env = environment.copy()
        
        # Execute the workflow until completion or failure
        while not workflow.is_workflow_completed():
            # Get the next steps to execute
            next_steps = workflow.get_next_steps()
            
            if not next_steps:
                # If there are no next steps but the workflow is not completed,
                # there might be a deadlock or all remaining steps have failed dependencies
                if workflow.in_progress_steps:
                    # There are steps in progress, wait for them to complete
                    self.logger.info("Waiting for in-progress steps to complete...")
                    time.sleep(1)
                    continue
                else:
                    # No steps in progress and no next steps, but workflow not completed
                    self.logger.error("Workflow execution stalled. Check for deadlocks or failed dependencies.")
                    break
            
            # Execute the next steps
            self._execute_steps(next_steps, workflow, env, invoke_agent_fn)
        
        # Log workflow completion status
        completed_steps = len(workflow.completed_steps)
        total_steps = len(workflow.steps)
        failed_steps = len(workflow.failed_steps)
        
        self.logger.info(f"Workflow execution completed: {completed_steps}/{total_steps} steps completed, "
                        f"{failed_steps} steps failed.")
        
        return env
    
    def _execute_steps(self, steps: List[WorkflowStep], workflow: WorkflowGraph, 
                      environment: Dict[str, Any], invoke_agent_fn: Callable) -> None:
        """
        Execute a batch of workflow steps.
        
        Args:
            steps: List of workflow steps to execute
            workflow: The workflow graph
            environment: The shared environment state
            invoke_agent_fn: Function to invoke an agent
        """
        # Limit the number of steps to execute in parallel
        steps_to_execute = steps[:self.max_parallel]
        
        # Mark steps as in progress
        for step in steps_to_execute:
            workflow.mark_step_in_progress(step.id)
            self.logger.info(f"Starting execution of step '{step.id}': {step.description}")
        
        # Execute steps in parallel using a thread pool
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = []
            for step in steps_to_execute:
                future = executor.submit(
                    self._execute_step, step, workflow, environment, invoke_agent_fn
                )
                futures.append((future, step))
            
            # Wait for all futures to complete
            for future, step in futures:
                try:
                    result = future.result()
                    # Update the environment with the step result
                    if result:
                        for key, value in result.items():
                            if key in step.output_keys:
                                # Handle nested keys with dot notation
                                if "." in key:
                                    parts = key.split(".")
                                    target = environment
                                    for part in parts[:-1]:
                                        if part not in target:
                                            target[part] = {}
                                        target = target[part]
                                    target[parts[-1]] = value
                                else:
                                    environment[key] = value
                    
                    # Mark the step as completed
                    workflow.mark_step_completed(step.id, result)
                    self.logger.info(f"Step '{step.id}' completed successfully")
                    
                except Exception as e:
                    self.logger.error(f"Step '{step.id}' failed: {str(e)}")
                    
                    # Check if we should retry the step
                    retry_count = self.retry_counts.get(step.id, 0)
                    if retry_count < self.max_retries:
                        self.retry_counts[step.id] = retry_count + 1
                        self.logger.info(f"Retrying step '{step.id}' (attempt {retry_count + 1}/{self.max_retries})")
                        workflow.mark_step_in_progress(step.id)
                        # Retry the step immediately
                        try:
                            result = self._execute_step(step, workflow, environment, invoke_agent_fn)
                            # Update the environment with the step result
                            if result:
                                for key, value in result.items():
                                    if key in step.output_keys:
                                        # Handle nested keys with dot notation
                                        if "." in key:
                                            parts = key.split(".")
                                            target = environment
                                            for part in parts[:-1]:
                                                if part not in target:
                                                    target[part] = {}
                                                target = target[part]
                                            target[parts[-1]] = value
                                        else:
                                            environment[key] = value
                            
                            # Mark the step as completed
                            workflow.mark_step_completed(step.id, result)
                            self.logger.info(f"Step '{step.id}' completed successfully on retry")
                        except Exception as retry_e:
                            self.logger.error(f"Step '{step.id}' failed on retry: {str(retry_e)}")
                            workflow.mark_step_failed(step.id, str(retry_e))
                    else:
                        workflow.mark_step_failed(step.id, str(e))
    
    def _execute_step(self, step: WorkflowStep, workflow: WorkflowGraph, 
                     environment: Dict[str, Any], invoke_agent_fn: Callable) -> Dict[str, Any]:
        """
        Execute a single workflow step.
        
        Args:
            step: The workflow step to execute
            workflow: The workflow graph
            environment: The shared environment state
            invoke_agent_fn: Function to invoke an agent
            
        Returns:
            Dict containing the step's results
            
        Raises:
            Exception: If the step execution fails
        """
        # Get the inputs for this step from the environment
        inputs = step.get_input_from_environment(environment)
        
        # Add the step information to the inputs
        inputs["step_id"] = step.id
        inputs["task"] = step.task
        inputs["description"] = step.description
        
        # Invoke the agent
        try:
            result = invoke_agent_fn(step.agent_name, **inputs)
            return result
        except Exception as e:
            self.logger.error(f"Error executing step '{step.id}': {str(e)}")
            raise
    
    async def execute_workflow_async(self, workflow: WorkflowGraph, environment: Dict[str, Any], 
                                   invoke_agent_fn: Callable) -> Dict[str, Any]:
        """
        Execute a workflow asynchronously.
        
        Args:
            workflow: The workflow graph to execute
            environment: The shared environment state
            invoke_agent_fn: Async function to invoke an agent
            
        Returns:
            Updated environment after workflow execution
        """
        # Create a copy of the environment to avoid modifying the original
        env = environment.copy()
        
        # Execute the workflow until completion or failure
        while not workflow.is_workflow_completed():
            # Get the next steps to execute
            next_steps = workflow.get_next_steps()
            
            if not next_steps:
                # If there are no next steps but the workflow is not completed,
                # there might be a deadlock or all remaining steps have failed dependencies
                if workflow.in_progress_steps:
                    # There are steps in progress, wait for them to complete
                    self.logger.info("Waiting for in-progress steps to complete...")
                    await asyncio.sleep(1)
                    continue
                else:
                    # No steps in progress and no next steps, but workflow not completed
                    self.logger.error("Workflow execution stalled. Check for deadlocks or failed dependencies.")
                    break
            
            # Execute the next steps
            await self._execute_steps_async(next_steps, workflow, env, invoke_agent_fn)
        
        # Log workflow completion status
        completed_steps = len(workflow.completed_steps)
        total_steps = len(workflow.steps)
        failed_steps = len(workflow.failed_steps)
        
        self.logger.info(f"Workflow execution completed: {completed_steps}/{total_steps} steps completed, "
                        f"{failed_steps} steps failed.")
        
        return env
    
    async def _execute_steps_async(self, steps: List[WorkflowStep], workflow: WorkflowGraph, 
                                 environment: Dict[str, Any], invoke_agent_fn: Callable) -> None:
        """
        Execute a batch of workflow steps asynchronously.
        
        Args:
            steps: List of workflow steps to execute
            workflow: The workflow graph
            environment: The shared environment state
            invoke_agent_fn: Async function to invoke an agent
        """
        # Limit the number of steps to execute in parallel
        steps_to_execute = steps[:self.max_parallel]
        
        # Mark steps as in progress
        for step in steps_to_execute:
            workflow.mark_step_in_progress(step.id)
            self.logger.info(f"Starting execution of step '{step.id}': {step.description}")
        
        # Execute steps in parallel
        tasks = []
        for step in steps_to_execute:
            task = asyncio.create_task(
                self._execute_step_async(step, workflow, environment, invoke_agent_fn)
            )
            tasks.append((task, step))
        
        # Wait for all tasks to complete
        for task, step in tasks:
            try:
                result = await task
                # Update the environment with the step result
                if result:
                    for key, value in result.items():
                        if key in step.output_keys:
                            # Handle nested keys with dot notation
                            if "." in key:
                                parts = key.split(".")
                                target = environment
                                for part in parts[:-1]:
                                    if part not in target:
                                        target[part] = {}
                                    target = target[part]
                                target[parts[-1]] = value
                            else:
                                environment[key] = value
                
                # Mark the step as completed
                workflow.mark_step_completed(step.id, result)
                self.logger.info(f"Step '{step.id}' completed successfully")
                
            except Exception as e:
                self.logger.error(f"Step '{step.id}' failed: {str(e)}")
                
                # Check if we should retry the step
                retry_count = self.retry_counts.get(step.id, 0)
                if retry_count < self.max_retries:
                    self.retry_counts[step.id] = retry_count + 1
                    self.logger.info(f"Retrying step '{step.id}' (attempt {retry_count + 1}/{self.max_retries})")
                    workflow.mark_step_in_progress(step.id)
                    # Retry the step immediately
                    try:
                        result = await self._execute_step_async(step, workflow, environment, invoke_agent_fn)
                        # Update the environment with the step result
                        if result:
                            for key, value in result.items():
                                if key in step.output_keys:
                                    # Handle nested keys with dot notation
                                    if "." in key:
                                        parts = key.split(".")
                                        target = environment
                                        for part in parts[:-1]:
                                            if part not in target:
                                                target[part] = {}
                                            target = target[part]
                                        target[parts[-1]] = value
                                    else:
                                        environment[key] = value
                        
                        # Mark the step as completed
                        workflow.mark_step_completed(step.id, result)
                        self.logger.info(f"Step '{step.id}' completed successfully on retry")
                    except Exception as retry_e:
                        self.logger.error(f"Step '{step.id}' failed on retry: {str(retry_e)}")
                        workflow.mark_step_failed(step.id, str(retry_e))
                else:
                    workflow.mark_step_failed(step.id, str(e))
    
    async def _execute_step_async(self, step: WorkflowStep, workflow: WorkflowGraph, 
                                environment: Dict[str, Any], invoke_agent_fn: Callable) -> Dict[str, Any]:
        """
        Execute a single workflow step asynchronously.
        
        Args:
            step: The workflow step to execute
            workflow: The workflow graph
            environment: The shared environment state
            invoke_agent_fn: Async function to invoke an agent
            
        Returns:
            Dict containing the step's results
            
        Raises:
            Exception: If the step execution fails
        """
        # Get the inputs for this step from the environment
        inputs = step.get_input_from_environment(environment)
        
        # Add the step information to the inputs
        inputs["step_id"] = step.id
        inputs["task"] = step.task
        inputs["description"] = step.description
        
        # Invoke the agent
        try:
            result = await invoke_agent_fn(step.agent_name, **inputs)
            return result
        except Exception as e:
            self.logger.error(f"Error executing step '{step.id}': {str(e)}")
            raise