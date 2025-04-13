"""
Supervisor agent for coordinating the agent swarm.
"""

import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime

from shared.state import SharedState, Task, TaskStatus, TaskType
from agents.base_agent import BaseAgent


class SupervisorAgent(BaseAgent):
    """Supervisor agent for coordinating the agent swarm."""

    def __init__(
        self,
        shared_state: SharedState,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-lite",
    ):
        """Initialize the supervisor agent.
        
        Args:
            shared_state: The shared state object.
            api_key: The Google AI Studio API key.
            model_name: The name of the model to use.
        """
        super().__init__(
            name="SupervisorAgent",
            description="Coordinates the work of specialized agents in the data pipeline.",
            shared_state=shared_state,
            api_key=api_key,
            model_name=model_name,
        )
        
        # Keep track of the specialized agents
        self.agents: Dict[str, BaseAgent] = {}
        
        # Keep track of the current pipeline state
        self.pipeline_state: str = "idle"
        self.current_task_id: Optional[str] = None

    def _register_tools(self) -> None:
        """Register the supervisor's tools."""
        self.register_tool(self.create_data_loading_task)
        self.register_tool(self.create_data_cleaning_task)
        self.register_tool(self.create_data_analysis_task)
        self.register_tool(self.create_data_visualization_task)
        self.register_tool(self.create_reporting_task)
        self.register_tool(self.get_pipeline_status)
        self.register_tool(self.get_task_status)

    def register_agent(self, agent: BaseAgent) -> None:
        """Register a specialized agent with the supervisor.
        
        Args:
            agent: The agent to register.
        """
        self.agents[agent.name] = agent

    def create_data_loading_task(self, description: str) -> Dict[str, Any]:
        """Create a data loading task.
        
        Args:
            description: A description of the task.
            
        Returns:
            Information about the created task.
        """
        task = self.create_task(TaskType.DATA_LOADING, description)
        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "description": task.description,
            "status": task.status,
        }

    def create_data_cleaning_task(self, description: str) -> Dict[str, Any]:
        """Create a data cleaning task.
        
        Args:
            description: A description of the task.
            
        Returns:
            Information about the created task.
        """
        task = self.create_task(TaskType.DATA_CLEANING, description)
        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "description": task.description,
            "status": task.status,
        }

    def create_data_analysis_task(self, description: str) -> Dict[str, Any]:
        """Create a data analysis task.
        
        Args:
            description: A description of the task.
            
        Returns:
            Information about the created task.
        """
        task = self.create_task(TaskType.DATA_ANALYSIS, description)
        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "description": task.description,
            "status": task.status,
        }

    def create_data_visualization_task(self, description: str) -> Dict[str, Any]:
        """Create a data visualization task.
        
        Args:
            description: A description of the task.
            
        Returns:
            Information about the created task.
        """
        task = self.create_task(TaskType.DATA_VISUALIZATION, description)
        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "description": task.description,
            "status": task.status,
        }

    def create_reporting_task(self, description: str) -> Dict[str, Any]:
        """Create a reporting task.
        
        Args:
            description: A description of the task.
            
        Returns:
            Information about the created task.
        """
        task = self.create_task(TaskType.REPORTING, description)
        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "description": task.description,
            "status": task.status,
        }

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get the current status of the pipeline.
        
        Returns:
            The current pipeline status.
        """
        return {
            "state": self.pipeline_state,
            "current_task_id": self.current_task_id,
            "pending_tasks": len(self.shared_state.get_pending_tasks()),
            "completed_tasks": len(self.shared_state.task_history),
            "errors": self.shared_state.errors,
            "warnings": self.shared_state.warnings,
        }

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a task.
        
        Args:
            task_id: The ID of the task.
            
        Returns:
            The task status.
        """
        task = self.shared_state.get_task(task_id)
        if task:
            return {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "description": task.description,
                "status": task.status,
                "assigned_to": task.assigned_to,
                "created_at": task.created_at,
                "completed_at": task.completed_at,
                "error": task.error,
            }
        
        # Check task history
        for task in self.shared_state.task_history:
            if task.task_id == task_id:
                return {
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "description": task.description,
                    "status": task.status,
                    "assigned_to": task.assigned_to,
                    "created_at": task.created_at,
                    "completed_at": task.completed_at,
                    "result": task.result,
                    "error": task.error,
                }
        
        return {"error": f"Task with ID {task_id} not found"}

    async def _process_task(self, task: Task) -> Dict[str, Any]:
        """Process a task.
        
        This method is called by the base agent's process_task method.
        
        Args:
            task: The task to process.
            
        Returns:
            The result of processing the task.
        """
        # Update the pipeline state
        self.pipeline_state = f"processing_{task.task_type}"
        self.current_task_id = task.task_id
        
        # Determine which agent should handle the task
        agent_name = None
        if task.task_type == TaskType.DATA_LOADING:
            agent_name = "DataLoadingAgent"
        elif task.task_type == TaskType.DATA_CLEANING:
            agent_name = "DataCleaningAgent"
        elif task.task_type == TaskType.DATA_ANALYSIS:
            agent_name = "DataAnalysisAgent"
        elif task.task_type == TaskType.DATA_VISUALIZATION:
            agent_name = "DataVisualizationAgent"
        elif task.task_type == TaskType.REPORTING:
            agent_name = "ReportingAgent"
        
        if agent_name and agent_name in self.agents:
            # Delegate the task to the appropriate agent
            agent = self.agents[agent_name]
            await agent.process_task(task)
            
            # Get the completed task from the history
            for completed_task in self.shared_state.task_history:
                if completed_task.task_id == task.task_id:
                    # Update the pipeline state
                    self.pipeline_state = "idle"
                    self.current_task_id = None
                    
                    return completed_task.result
        
        # If we get here, something went wrong
        self.pipeline_state = "idle"
        self.current_task_id = None
        
        return {"error": f"Failed to process task {task.task_id}"}

    async def plan_pipeline(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Plan the data pipeline based on the uploaded file.
        
        Args:
            file_info: Information about the uploaded file.
            
        Returns:
            The pipeline plan.
        """
        # Update the shared state with file information
        self.shared_state.filename = file_info.get("filename", "")
        self.shared_state.file_extension = file_info.get("file_extension", "")
        self.shared_state.file_size = file_info.get("file_size", 0)
        
        # Create a prompt for the model to plan the pipeline
        prompt = f"""
        You are the supervisor agent for a data analytics pipeline. You need to plan the pipeline for a new file.
        
        File information:
        - Filename: {self.shared_state.filename}
        - File extension: {self.shared_state.file_extension}
        - File size: {self.shared_state.file_size} bytes
        
        Your task is to create a plan for processing this file through the data pipeline.
        The pipeline consists of the following stages:
        1. Data Loading: Loading and parsing the file
        2. Data Cleaning: Cleaning and preprocessing the data
        3. Data Analysis: Analyzing the data and generating insights
        4. Data Visualization: Creating visualizations based on the data
        5. Reporting: Compiling insights and visualizations into a coherent report
        
        For each stage, create a task with a detailed description of what needs to be done.
        """
        
        # Run the model to generate a plan
        response = await self.run(prompt)
        
        # Create tasks based on the model's response
        # For now, we'll create a simple linear pipeline
        tasks = []
        
        # Data Loading
        loading_task = self.create_task(
            TaskType.DATA_LOADING,
            f"Load and parse the file {self.shared_state.filename}"
        )
        tasks.append(loading_task)
        
        # Data Cleaning
        cleaning_task = self.create_task(
            TaskType.DATA_CLEANING,
            "Clean and preprocess the data"
        )
        tasks.append(cleaning_task)
        
        # Data Analysis
        analysis_task = self.create_task(
            TaskType.DATA_ANALYSIS,
            "Analyze the data and generate insights"
        )
        tasks.append(analysis_task)
        
        # Data Visualization
        visualization_task = self.create_task(
            TaskType.DATA_VISUALIZATION,
            "Create visualizations based on the data"
        )
        tasks.append(visualization_task)
        
        # Reporting
        reporting_task = self.create_task(
            TaskType.REPORTING,
            "Compile insights and visualizations into a coherent report"
        )
        tasks.append(reporting_task)
        
        return {
            "plan": {
                "file_info": {
                    "filename": self.shared_state.filename,
                    "file_extension": self.shared_state.file_extension,
                    "file_size": self.shared_state.file_size,
                },
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "description": task.description,
                    }
                    for task in tasks
                ],
                "model_response": response.get("text", ""),
            }
        }

    async def execute_pipeline(self) -> Dict[str, Any]:
        """Execute the data pipeline.
        
        Returns:
            The results of the pipeline execution.
        """
        results = []
        
        # Get all pending tasks
        tasks = self.shared_state.get_pending_tasks()
        
        # Sort tasks by type to ensure they're processed in the right order
        task_type_order = {
            TaskType.DATA_LOADING: 0,
            TaskType.DATA_CLEANING: 1,
            TaskType.DATA_ANALYSIS: 2,
            TaskType.DATA_VISUALIZATION: 3,
            TaskType.REPORTING: 4,
        }
        tasks.sort(key=lambda task: task_type_order.get(task.task_type, 999))
        
        # Process each task
        for task in tasks:
            try:
                # Process the task
                result = await self.process_task(task)
                
                # Add the result to the list
                results.append({
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "status": "completed",
                    "result": result,
                })
            except Exception as e:
                # Add the error to the list
                results.append({
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "status": "failed",
                    "error": str(e),
                })
                
                # Stop processing if a task fails
                break
        
        return {
            "pipeline_results": results,
            "final_state": self.shared_state.to_dict(),
        }