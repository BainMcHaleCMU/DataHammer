"""
Base agent class for the agent swarm.
"""

import os
import json
import uuid
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import google.generativeai as genai
from pydantic import BaseModel, Field

from shared.state import SharedState, Task, TaskStatus, TaskType


class FunctionDeclaration(BaseModel):
    """Function declaration for the agent."""
    name: str
    description: str
    parameters: Dict[str, Any]


class BaseAgent:
    """Base agent class for the agent swarm."""

    def __init__(
        self,
        name: str,
        description: str,
        shared_state: SharedState,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-lite",
    ):
        """Initialize the agent.
        
        Args:
            name: The name of the agent.
            description: A description of the agent's purpose.
            shared_state: The shared state object.
            api_key: The Google AI Studio API key.
            model_name: The name of the model to use.
        """
        self.name = name
        self.description = description
        self.shared_state = shared_state
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.model_name = model_name or os.environ.get("MODEL_NAME", "gemini-2.0-flash-lite")
        self.tools: List[Dict[str, Any]] = []
        self.function_map: Dict[str, Callable] = {}
        
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        
        # Register the agent's tools
        self._register_tools()

    def _register_tools(self) -> None:
        """Register the agent's tools.
        
        This method should be overridden by subclasses to register their tools.
        """
        pass

    def register_tool(self, func: Callable) -> None:
        """Register a tool with the agent.
        
        Args:
            func: The function to register as a tool.
        """
        # Extract function name, docstring, and type hints
        name = func.__name__
        docstring = func.__doc__ or ""
        annotations = func.__annotations__
        
        # Create a function declaration
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        # Add parameters based on type hints
        for param_name, param_type in annotations.items():
            if param_name == "return":
                continue
                
            # Determine the JSON schema type based on the Python type
            if param_type == str:
                json_type = "string"
            elif param_type == int:
                json_type = "integer"
            elif param_type == float:
                json_type = "number"
            elif param_type == bool:
                json_type = "boolean"
            elif param_type == list or param_type == List:
                json_type = "array"
            elif param_type == dict or param_type == Dict:
                json_type = "object"
            else:
                json_type = "string"  # Default to string for complex types
                
            # Add the parameter to the schema
            parameters["properties"][param_name] = {
                "type": json_type,
                "description": f"Parameter {param_name}"
            }
            parameters["required"].append(param_name)
            
        # Create the function declaration
        function_declaration = {
            "name": name,
            "description": docstring,
            "parameters": parameters
        }
        
        # Add the function to the tools list
        self.tools.append({"function_declarations": [function_declaration]})
        
        # Add the function to the function map
        self.function_map[name] = func

    def create_task(self, task_type: TaskType, description: str) -> Task:
        """Create a new task.
        
        Args:
            task_type: The type of task.
            description: A description of the task.
            
        Returns:
            The created task.
        """
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            task_type=task_type,
            description=description,
            assigned_to=self.name,
            created_at=datetime.now().isoformat(),
            status=TaskStatus.PENDING
        )
        self.shared_state.add_task(task)
        return task

    async def process_task(self, task: Task) -> None:
        """Process a task.
        
        Args:
            task: The task to process.
        """
        # Mark the task as in progress
        self.shared_state.update_task(task.task_id, status=TaskStatus.IN_PROGRESS)
        
        try:
            # Process the task
            result = await self._process_task(task)
            
            # Mark the task as completed
            self.shared_state.complete_task(task.task_id, result)
        except Exception as e:
            # Mark the task as failed
            self.shared_state.fail_task(task.task_id, str(e))
            raise

    async def _process_task(self, task: Task) -> Dict[str, Any]:
        """Process a task.
        
        This method should be overridden by subclasses to implement their task processing logic.
        
        Args:
            task: The task to process.
            
        Returns:
            The result of processing the task.
        """
        raise NotImplementedError("Subclasses must implement _process_task")

    async def run(self, prompt: str, tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Run the agent with a prompt.
        
        Args:
            prompt: The prompt to send to the model.
            tools: Optional list of tools to use for this run.
            
        Returns:
            The response from the model.
        """
        tools_to_use = tools or self.tools
        
        # Create the generation config
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        # Create the safety settings
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        # Send the prompt to the model
        response = self.model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=tools_to_use
        )
        
        # Check if the response contains a function call
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                content = candidate.content
                if hasattr(content, 'parts') and content.parts:
                    part = content.parts[0]
                    if hasattr(part, 'function_call') and part.function_call:
                        function_call = part.function_call
                        function_name = function_call.name
                        function_args = function_call.args
                        
                        # Call the function
                        if function_name in self.function_map:
                            function = self.function_map[function_name]
                            result = function(**function_args)
                            
                            # Create a response with the function result
                            return {
                                "function_call": {
                                    "name": function_name,
                                    "args": function_args
                                },
                                "function_response": result,
                                "text": response.text
                            }
        
        # If no function call, just return the text
        return {"text": response.text}

    def __str__(self) -> str:
        """Return a string representation of the agent."""
        return f"{self.name}: {self.description}"