"""
Workflow Planner Module

This module defines the WorkflowPlanner class that dynamically plans workflows based on goals.
"""

from typing import Any, Dict, List, Optional
import uuid
from llama_index.core.llms import LLM
from llama_index.core.settings import Settings

from .workflow_step import WorkflowStep
from .workflow_graph import WorkflowGraph


class WorkflowPlanner:
    """
    Dynamically plans workflows based on user goals and data characteristics.
    
    The WorkflowPlanner is responsible for:
    - Analyzing user goals
    - Creating a workflow graph of steps
    - Determining the optimal sequence of agent invocations
    - Adapting the workflow based on intermediate results
    """
    
    def __init__(self, llm: Optional[LLM] = None):
        """
        Initialize the WorkflowPlanner.
        
        Args:
            llm: Optional language model to use for planning
        """
        self.llm = llm or Settings.llm
        if not self.llm:
            raise ValueError("No language model available. Please configure Settings.llm or provide an LLM.")
    
    def create_workflow(self, goals: List[str], environment: Dict[str, Any]) -> WorkflowGraph:
        """
        Create a workflow based on user goals and the current environment.
        
        Args:
            goals: List of user-defined goals
            environment: The shared environment state
            
        Returns:
            A WorkflowGraph containing the planned steps
        """
        # Create an empty workflow graph
        workflow = WorkflowGraph()
        
        # Generate a plan based on the goals and environment
        plan = self._generate_plan(goals, environment)
        
        # Convert the plan into workflow steps
        for step_data in plan:
            step = WorkflowStep(
                id=step_data.get("id", f"step_{uuid.uuid4().hex[:8]}"),
                agent_name=step_data["agent_name"],
                task=step_data["task"],
                description=step_data["description"],
                input_keys=step_data.get("input_keys", []),
                output_keys=step_data.get("output_keys", []),
                dependencies=step_data.get("dependencies", []),
                parameters=step_data.get("parameters", {})
            )
            workflow.add_step(step)
        
        # Validate the workflow
        workflow.validate()
        
        return workflow
    
    def _generate_plan(self, goals: List[str], environment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate a plan based on user goals and the current environment.
        
        Args:
            goals: List of user-defined goals
            environment: The shared environment state
            
        Returns:
            List of step definitions
        """
        # Get available agents
        available_agents = environment.get("Available Agents", {})
        
        # Prepare the prompt for the LLM
        prompt = self._create_planning_prompt(goals, available_agents, environment)
        
        # Generate the plan using the LLM
        response = self.llm.complete(prompt)
        
        # Parse the response into a structured plan
        plan = self._parse_plan_response(response.text)
        
        return plan
    
    def _create_planning_prompt(self, goals: List[str], available_agents: Dict[str, Any], 
                               environment: Dict[str, Any]) -> str:
        """
        Create a prompt for the LLM to generate a workflow plan.
        
        Args:
            goals: List of user-defined goals
            available_agents: Dict of available agents
            environment: The shared environment state
            
        Returns:
            Prompt string for the LLM
        """
        goals_str = "\n".join([f"- {goal}" for goal in goals])
        agents_str = "\n".join([f"- {name}: {info.get('description', 'No description')}" 
                               for name, info in available_agents.items()])
        
        # Include relevant environment information
        data_sources = environment.get("Data", {})
        data_sources_str = "\n".join([f"- {name}: {source}" for name, source in data_sources.items()])
        
        prompt = f"""
        You are an AI workflow planner for a data science project. Your task is to create a detailed workflow plan
        based on the following goals and available agents.
        
        GOALS:
        {goals_str}
        
        AVAILABLE AGENTS:
        {agents_str}
        
        DATA SOURCES:
        {data_sources_str}
        
        Create a workflow plan with the following structure:
        1. Break down the goals into a sequence of steps
        2. For each step, specify:
           - A unique ID
           - The agent responsible for executing the step
           - The specific task to be performed
           - A clear description of what the step accomplishes
           - Input requirements (keys from the environment)
           - Expected outputs (keys to be added/updated in the environment)
           - Dependencies on other steps (IDs of steps that must be completed first)
           - Any additional parameters needed
        
        The workflow should be logical, efficient, and aligned with standard data science practices.
        Return the plan as a JSON array of step objects.
        
        Example step format:
        {{
            "id": "data_loading",
            "agent_name": "DataLoadingAgent",
            "task": "load_csv_data",
            "description": "Load the CSV data from the provided source",
            "input_keys": ["Data.csv_file"],
            "output_keys": ["Data Overview.raw_data", "Data Overview.schema"],
            "dependencies": [],
            "parameters": {{"delimiter": ","}}
        }}
        """
        
        return prompt
    
    def _parse_plan_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse the LLM response into a structured plan.
        
        Args:
            response: The LLM response text
            
        Returns:
            List of step definitions
        """
        import json
        import re
        
        # Extract JSON array from the response
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Fallback to a simple plan if JSON parsing fails
                return self._create_fallback_plan()
        
        # Fallback to a simple plan if no JSON array is found
        return self._create_fallback_plan()
    
    def _create_fallback_plan(self) -> List[Dict[str, Any]]:
        """
        Create a fallback plan when LLM planning fails.
        
        Returns:
            List of step definitions for a basic data science workflow
        """
        return [
            {
                "id": "data_loading",
                "agent_name": "DataLoadingAgent",
                "task": "load_data",
                "description": "Load data from the provided sources",
                "input_keys": ["Data"],
                "output_keys": ["Data Overview.raw_data"],
                "dependencies": [],
                "parameters": {}
            },
            {
                "id": "data_exploration",
                "agent_name": "ExplorationAgent",
                "task": "explore_data",
                "description": "Explore and analyze the loaded data",
                "input_keys": ["Data Overview.raw_data"],
                "output_keys": ["Data Overview.summary", "Data Overview.statistics"],
                "dependencies": ["data_loading"],
                "parameters": {}
            },
            {
                "id": "data_cleaning",
                "agent_name": "CleaningAgent",
                "task": "clean_data",
                "description": "Clean and preprocess the data",
                "input_keys": ["Data Overview.raw_data", "Data Overview.summary"],
                "output_keys": ["Cleaned Data.processed_data"],
                "dependencies": ["data_exploration"],
                "parameters": {}
            },
            {
                "id": "data_analysis",
                "agent_name": "AnalysisAgent",
                "task": "analyze_data",
                "description": "Perform in-depth analysis on the cleaned data",
                "input_keys": ["Cleaned Data.processed_data"],
                "output_keys": ["Analysis Results.insights"],
                "dependencies": ["data_cleaning"],
                "parameters": {}
            },
            {
                "id": "modeling",
                "agent_name": "ModelingAgent",
                "task": "build_model",
                "description": "Build and train predictive models",
                "input_keys": ["Cleaned Data.processed_data"],
                "output_keys": ["Models.trained_model", "Models.performance"],
                "dependencies": ["data_cleaning"],
                "parameters": {}
            },
            {
                "id": "visualization",
                "agent_name": "VisualizationAgent",
                "task": "create_visualizations",
                "description": "Create visualizations of the data and results",
                "input_keys": ["Cleaned Data.processed_data", "Analysis Results.insights", "Models.performance"],
                "output_keys": ["Visualizations.plots"],
                "dependencies": ["data_analysis", "modeling"],
                "parameters": {}
            },
            {
                "id": "reporting",
                "agent_name": "ReportingAgent",
                "task": "generate_report",
                "description": "Generate a comprehensive report of findings",
                "input_keys": ["Analysis Results.insights", "Models.performance", "Visualizations.plots"],
                "output_keys": ["JupyterLogbook"],
                "dependencies": ["visualization"],
                "parameters": {}
            }
        ]
    
    def update_workflow(self, workflow: WorkflowGraph, environment: Dict[str, Any]) -> WorkflowGraph:
        """
        Update an existing workflow based on intermediate results.
        
        Args:
            workflow: The current workflow graph
            environment: The updated environment state
            
        Returns:
            Updated WorkflowGraph
        """
        # Analyze the current state and results
        completed_steps = workflow.completed_steps
        failed_steps = workflow.failed_steps
        
        # If there are failed steps, generate recovery steps
        if failed_steps:
            recovery_plan = self._generate_recovery_plan(workflow, environment)
            
            # Add recovery steps to the workflow
            for step_data in recovery_plan:
                if step_data["id"] not in workflow.steps:
                    step = WorkflowStep(
                        id=step_data["id"],
                        agent_name=step_data["agent_name"],
                        task=step_data["task"],
                        description=step_data["description"],
                        input_keys=step_data.get("input_keys", []),
                        output_keys=step_data.get("output_keys", []),
                        dependencies=step_data.get("dependencies", []),
                        parameters=step_data.get("parameters", {})
                    )
                    workflow.add_step(step)
        
        # Validate the updated workflow
        workflow.validate()
        
        return workflow
    
    def _generate_recovery_plan(self, workflow: WorkflowGraph, 
                               environment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate a recovery plan for failed steps.
        
        Args:
            workflow: The current workflow graph
            environment: The shared environment state
            
        Returns:
            List of recovery step definitions
        """
        # Get failed steps
        failed_steps = [workflow.steps[step_id] for step_id in workflow.failed_steps]
        
        # Create a recovery plan
        recovery_plan = []
        
        for failed_step in failed_steps:
            # Create a recovery step ID
            recovery_id = f"recovery_{failed_step.id}"
            
            # Create a recovery step
            recovery_step = {
                "id": recovery_id,
                "agent_name": failed_step.agent_name,
                "task": f"retry_{failed_step.task}",
                "description": f"Retry failed step: {failed_step.description}",
                "input_keys": failed_step.input_keys,
                "output_keys": failed_step.output_keys,
                "dependencies": failed_step.dependencies,
                "parameters": {
                    **failed_step.parameters,
                    "is_recovery": True,
                    "original_step_id": failed_step.id,
                    "error": failed_step.result.get("error") if failed_step.result else None
                }
            }
            
            recovery_plan.append(recovery_step)
            
            # Update dependencies of steps that depended on the failed step
            for step in workflow.get_all_steps():
                if failed_step.id in step.dependencies:
                    # Create a new step with updated dependencies
                    updated_step = {
                        "id": step.id,
                        "agent_name": step.agent_name,
                        "task": step.task,
                        "description": step.description,
                        "input_keys": step.input_keys,
                        "output_keys": step.output_keys,
                        "dependencies": [recovery_id if dep == failed_step.id else dep for dep in step.dependencies],
                        "parameters": step.parameters
                    }
                    
                    recovery_plan.append(updated_step)
        
        return recovery_plan