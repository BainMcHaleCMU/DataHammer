"""
Data loading agent for the agent swarm.
"""

import io
import pandas as pd
from typing import Dict, List, Any, Optional

from shared.state import SharedState, Task, Insight
from agents.base_agent import BaseAgent


class DataLoadingAgent(BaseAgent):
    """Data loading agent for the agent swarm."""

    def __init__(
        self,
        shared_state: SharedState,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-lite",
    ):
        """Initialize the data loading agent.
        
        Args:
            shared_state: The shared state object.
            api_key: The Google AI Studio API key.
            model_name: The name of the model to use.
        """
        super().__init__(
            name="DataLoadingAgent",
            description="Loads and parses data files.",
            shared_state=shared_state,
            api_key=api_key,
            model_name=model_name,
        )

    def _register_tools(self) -> None:
        """Register the agent's tools."""
        self.register_tool(self.load_csv)
        self.register_tool(self.load_excel)
        self.register_tool(self.get_dataset_info)
        self.register_tool(self.add_dataset_insight)

    def load_csv(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Load a CSV file into a pandas DataFrame.
        
        Args:
            file_content: The content of the file as bytes.
            filename: The name of the file.
            
        Returns:
            Information about the loaded dataset.
        """
        try:
            # Load the CSV file
            df = pd.read_csv(io.BytesIO(file_content))
            
            # Store the DataFrame in the shared state
            self.shared_state.dataframe = df
            self.shared_state.filename = filename
            self.shared_state.file_extension = filename.split('.')[-1]
            self.shared_state.file_size = len(file_content)
            
            # Update the dataset info
            self._update_dataset_info()
            
            return {
                "success": True,
                "message": f"Successfully loaded {filename}",
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
            }
        except Exception as e:
            self.shared_state.add_error(f"Failed to load CSV file: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to load CSV file: {str(e)}",
            }

    def load_excel(self, file_content: bytes, filename: str, sheet_name: Optional[str] = None) -> Dict[str, Any]:
        """Load an Excel file into a pandas DataFrame.
        
        Args:
            file_content: The content of the file as bytes.
            filename: The name of the file.
            sheet_name: The name of the sheet to load. If None, the first sheet is loaded.
            
        Returns:
            Information about the loaded dataset.
        """
        try:
            # Load the Excel file
            if sheet_name:
                df = pd.read_excel(io.BytesIO(file_content), sheet_name=sheet_name)
            else:
                df = pd.read_excel(io.BytesIO(file_content))
            
            # Store the DataFrame in the shared state
            self.shared_state.dataframe = df
            self.shared_state.filename = filename
            self.shared_state.file_extension = filename.split('.')[-1]
            self.shared_state.file_size = len(file_content)
            
            # Update the dataset info
            self._update_dataset_info()
            
            return {
                "success": True,
                "message": f"Successfully loaded {filename}",
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
            }
        except Exception as e:
            self.shared_state.add_error(f"Failed to load Excel file: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to load Excel file: {str(e)}",
            }

    def _update_dataset_info(self) -> None:
        """Update the dataset info in the shared state."""
        df = self.shared_state.dataframe
        if df is not None:
            # Update the dataset info
            self.shared_state.dataset_info.rows = len(df)
            self.shared_state.dataset_info.columns = len(df.columns)
            self.shared_state.dataset_info.column_names = df.columns.tolist()
            
            # Update the dtypes
            dtypes = {}
            for column, dtype in df.dtypes.items():
                dtypes[column] = str(dtype)
            self.shared_state.dataset_info.dtypes = dtypes
            
            # Update the missing values
            missing_values = {}
            for column in df.columns:
                missing_values[column] = df[column].isna().sum()
            self.shared_state.dataset_info.missing_values = missing_values

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset.
        
        Returns:
            Information about the dataset.
        """
        if self.shared_state.dataframe is None:
            return {
                "success": False,
                "message": "No dataset has been loaded yet.",
            }
        
        return {
            "success": True,
            "dataset_info": self.shared_state.dataset_info.model_dump(),
        }

    def add_dataset_insight(self, text: str, importance: int = 1, category: str = "", related_columns: List[str] = None) -> Dict[str, Any]:
        """Add an insight about the dataset.
        
        Args:
            text: The text of the insight.
            importance: The importance of the insight (1-5).
            category: The category of the insight.
            related_columns: The columns related to the insight.
            
        Returns:
            Information about the added insight.
        """
        if related_columns is None:
            related_columns = []
            
        insight = Insight(
            text=text,
            importance=importance,
            category=category,
            related_columns=related_columns,
        )
        
        self.shared_state.add_insight(insight)
        
        return {
            "success": True,
            "message": "Insight added successfully.",
            "insight": insight.model_dump(),
        }

    async def _process_task(self, task: Task) -> Dict[str, Any]:
        """Process a data loading task.
        
        Args:
            task: The task to process.
            
        Returns:
            The result of processing the task.
        """
        # Check if we have a dataframe already
        if self.shared_state.dataframe is not None:
            # We already have a dataframe, so just return the dataset info
            return {
                "success": True,
                "message": "Dataset already loaded.",
                "dataset_info": self.shared_state.dataset_info.model_dump(),
            }
        
        # We don't have a dataframe, so we need to prompt the user to upload a file
        # This will be handled by the API endpoint
        return {
            "success": False,
            "message": "No dataset has been loaded yet. Please upload a file.",
        }