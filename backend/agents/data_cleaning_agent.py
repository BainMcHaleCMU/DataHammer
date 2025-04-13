"""
Data cleaning agent for the agent swarm.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union

from shared.state import SharedState, Task, Insight
from agents.base_agent import BaseAgent


class DataCleaningAgent(BaseAgent):
    """Data cleaning agent for the agent swarm."""

    def __init__(
        self,
        shared_state: SharedState,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-lite",
    ):
        """Initialize the data cleaning agent.
        
        Args:
            shared_state: The shared state object.
            api_key: The Google AI Studio API key.
            model_name: The name of the model to use.
        """
        super().__init__(
            name="DataCleaningAgent",
            description="Cleans and preprocesses data.",
            shared_state=shared_state,
            api_key=api_key,
            model_name=model_name,
        )

    def _register_tools(self) -> None:
        """Register the agent's tools."""
        self.register_tool(self.handle_missing_values)
        self.register_tool(self.remove_duplicates)
        self.register_tool(self.convert_column_type)
        self.register_tool(self.normalize_column)
        self.register_tool(self.drop_columns)
        self.register_tool(self.rename_columns)
        self.register_tool(self.add_cleaning_insight)
        self.register_tool(self.get_column_statistics)

    def handle_missing_values(self, column: str, strategy: str, value: Optional[Union[str, int, float]] = None) -> Dict[str, Any]:
        """Handle missing values in a column.
        
        Args:
            column: The name of the column.
            strategy: The strategy to use ('drop', 'fill_value', 'fill_mean', 'fill_median', 'fill_mode').
            value: The value to use if strategy is 'fill_value'.
            
        Returns:
            Information about the operation.
        """
        if self.shared_state.dataframe is None:
            return {
                "success": False,
                "message": "No dataset has been loaded yet.",
            }
        
        df = self.shared_state.dataframe
        
        if column not in df.columns:
            return {
                "success": False,
                "message": f"Column '{column}' not found in the dataset.",
            }
        
        try:
            if strategy == 'drop':
                # Drop rows with missing values in the column
                df = df.dropna(subset=[column])
            elif strategy == 'fill_value':
                # Fill missing values with a specific value
                if value is None:
                    return {
                        "success": False,
                        "message": "Value must be provided for 'fill_value' strategy.",
                    }
                df[column] = df[column].fillna(value)
            elif strategy == 'fill_mean':
                # Fill missing values with the mean
                if not pd.api.types.is_numeric_dtype(df[column]):
                    return {
                        "success": False,
                        "message": f"Column '{column}' is not numeric. Cannot use 'fill_mean' strategy.",
                    }
                df[column] = df[column].fillna(df[column].mean())
            elif strategy == 'fill_median':
                # Fill missing values with the median
                if not pd.api.types.is_numeric_dtype(df[column]):
                    return {
                        "success": False,
                        "message": f"Column '{column}' is not numeric. Cannot use 'fill_median' strategy.",
                    }
                df[column] = df[column].fillna(df[column].median())
            elif strategy == 'fill_mode':
                # Fill missing values with the mode
                df[column] = df[column].fillna(df[column].mode()[0])
            else:
                return {
                    "success": False,
                    "message": f"Unknown strategy '{strategy}'. Valid strategies are 'drop', 'fill_value', 'fill_mean', 'fill_median', 'fill_mode'.",
                }
            
            # Update the dataframe in the shared state
            self.shared_state.dataframe = df
            
            # Update the dataset info
            self._update_dataset_info()
            
            return {
                "success": True,
                "message": f"Successfully handled missing values in column '{column}' using strategy '{strategy}'.",
                "rows_before": len(self.shared_state.dataframe),
                "rows_after": len(df),
                "missing_values_before": self.shared_state.dataset_info.missing_values.get(column, 0),
                "missing_values_after": df[column].isna().sum(),
            }
        except Exception as e:
            self.shared_state.add_error(f"Failed to handle missing values: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to handle missing values: {str(e)}",
            }

    def remove_duplicates(self, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Remove duplicate rows from the dataset.
        
        Args:
            columns: The columns to consider when identifying duplicates. If None, all columns are used.
            
        Returns:
            Information about the operation.
        """
        if self.shared_state.dataframe is None:
            return {
                "success": False,
                "message": "No dataset has been loaded yet.",
            }
        
        df = self.shared_state.dataframe
        rows_before = len(df)
        
        try:
            if columns:
                # Check if all columns exist
                for column in columns:
                    if column not in df.columns:
                        return {
                            "success": False,
                            "message": f"Column '{column}' not found in the dataset.",
                        }
                
                # Remove duplicates based on the specified columns
                df = df.drop_duplicates(subset=columns)
            else:
                # Remove duplicates based on all columns
                df = df.drop_duplicates()
            
            # Update the dataframe in the shared state
            self.shared_state.dataframe = df
            
            # Update the dataset info
            self._update_dataset_info()
            
            return {
                "success": True,
                "message": f"Successfully removed duplicate rows.",
                "rows_before": rows_before,
                "rows_after": len(df),
                "duplicates_removed": rows_before - len(df),
            }
        except Exception as e:
            self.shared_state.add_error(f"Failed to remove duplicates: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to remove duplicates: {str(e)}",
            }

    def convert_column_type(self, column: str, target_type: str) -> Dict[str, Any]:
        """Convert a column to a different data type.
        
        Args:
            column: The name of the column.
            target_type: The target data type ('int', 'float', 'str', 'bool', 'datetime').
            
        Returns:
            Information about the operation.
        """
        if self.shared_state.dataframe is None:
            return {
                "success": False,
                "message": "No dataset has been loaded yet.",
            }
        
        df = self.shared_state.dataframe
        
        if column not in df.columns:
            return {
                "success": False,
                "message": f"Column '{column}' not found in the dataset.",
            }
        
        try:
            # Get the current type
            current_type = str(df[column].dtype)
            
            # Convert the column
            if target_type == 'int':
                df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
            elif target_type == 'float':
                df[column] = pd.to_numeric(df[column], errors='coerce')
            elif target_type == 'str':
                df[column] = df[column].astype(str)
            elif target_type == 'bool':
                df[column] = df[column].astype(bool)
            elif target_type == 'datetime':
                df[column] = pd.to_datetime(df[column], errors='coerce')
            else:
                return {
                    "success": False,
                    "message": f"Unknown target type '{target_type}'. Valid types are 'int', 'float', 'str', 'bool', 'datetime'.",
                }
            
            # Update the dataframe in the shared state
            self.shared_state.dataframe = df
            
            # Update the dataset info
            self._update_dataset_info()
            
            return {
                "success": True,
                "message": f"Successfully converted column '{column}' from '{current_type}' to '{target_type}'.",
                "column": column,
                "previous_type": current_type,
                "new_type": str(df[column].dtype),
            }
        except Exception as e:
            self.shared_state.add_error(f"Failed to convert column type: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to convert column type: {str(e)}",
            }

    def normalize_column(self, column: str, method: str = 'minmax') -> Dict[str, Any]:
        """Normalize a numeric column.
        
        Args:
            column: The name of the column.
            method: The normalization method ('minmax', 'zscore').
            
        Returns:
            Information about the operation.
        """
        if self.shared_state.dataframe is None:
            return {
                "success": False,
                "message": "No dataset has been loaded yet.",
            }
        
        df = self.shared_state.dataframe
        
        if column not in df.columns:
            return {
                "success": False,
                "message": f"Column '{column}' not found in the dataset.",
            }
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            return {
                "success": False,
                "message": f"Column '{column}' is not numeric. Cannot normalize.",
            }
        
        try:
            if method == 'minmax':
                # Min-max normalization
                min_val = df[column].min()
                max_val = df[column].max()
                df[column] = (df[column] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                # Z-score normalization
                mean = df[column].mean()
                std = df[column].std()
                df[column] = (df[column] - mean) / std
            else:
                return {
                    "success": False,
                    "message": f"Unknown normalization method '{method}'. Valid methods are 'minmax', 'zscore'.",
                }
            
            # Update the dataframe in the shared state
            self.shared_state.dataframe = df
            
            # Update the dataset info
            self._update_dataset_info()
            
            return {
                "success": True,
                "message": f"Successfully normalized column '{column}' using method '{method}'.",
                "column": column,
                "method": method,
                "min": float(df[column].min()),
                "max": float(df[column].max()),
                "mean": float(df[column].mean()),
                "std": float(df[column].std()),
            }
        except Exception as e:
            self.shared_state.add_error(f"Failed to normalize column: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to normalize column: {str(e)}",
            }

    def drop_columns(self, columns: List[str]) -> Dict[str, Any]:
        """Drop columns from the dataset.
        
        Args:
            columns: The names of the columns to drop.
            
        Returns:
            Information about the operation.
        """
        if self.shared_state.dataframe is None:
            return {
                "success": False,
                "message": "No dataset has been loaded yet.",
            }
        
        df = self.shared_state.dataframe
        
        # Check if all columns exist
        for column in columns:
            if column not in df.columns:
                return {
                    "success": False,
                    "message": f"Column '{column}' not found in the dataset.",
                }
        
        try:
            # Drop the columns
            df = df.drop(columns=columns)
            
            # Update the dataframe in the shared state
            self.shared_state.dataframe = df
            
            # Update the dataset info
            self._update_dataset_info()
            
            return {
                "success": True,
                "message": f"Successfully dropped columns: {', '.join(columns)}.",
                "columns_before": len(self.shared_state.dataset_info.column_names) + len(columns),
                "columns_after": len(self.shared_state.dataset_info.column_names),
                "dropped_columns": columns,
            }
        except Exception as e:
            self.shared_state.add_error(f"Failed to drop columns: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to drop columns: {str(e)}",
            }

    def rename_columns(self, column_mapping: Dict[str, str]) -> Dict[str, Any]:
        """Rename columns in the dataset.
        
        Args:
            column_mapping: A mapping from old column names to new column names.
            
        Returns:
            Information about the operation.
        """
        if self.shared_state.dataframe is None:
            return {
                "success": False,
                "message": "No dataset has been loaded yet.",
            }
        
        df = self.shared_state.dataframe
        
        # Check if all columns exist
        for old_name in column_mapping.keys():
            if old_name not in df.columns:
                return {
                    "success": False,
                    "message": f"Column '{old_name}' not found in the dataset.",
                }
        
        try:
            # Rename the columns
            df = df.rename(columns=column_mapping)
            
            # Update the dataframe in the shared state
            self.shared_state.dataframe = df
            
            # Update the dataset info
            self._update_dataset_info()
            
            return {
                "success": True,
                "message": f"Successfully renamed columns.",
                "renamed_columns": column_mapping,
            }
        except Exception as e:
            self.shared_state.add_error(f"Failed to rename columns: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to rename columns: {str(e)}",
            }

    def add_cleaning_insight(self, text: str, importance: int = 1, category: str = "", related_columns: List[str] = None) -> Dict[str, Any]:
        """Add an insight about the data cleaning process.
        
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

    def get_column_statistics(self, column: str) -> Dict[str, Any]:
        """Get statistics for a column.
        
        Args:
            column: The name of the column.
            
        Returns:
            Statistics for the column.
        """
        if self.shared_state.dataframe is None:
            return {
                "success": False,
                "message": "No dataset has been loaded yet.",
            }
        
        df = self.shared_state.dataframe
        
        if column not in df.columns:
            return {
                "success": False,
                "message": f"Column '{column}' not found in the dataset.",
            }
        
        try:
            # Get the column statistics
            stats = {}
            
            # Basic statistics
            stats["count"] = int(df[column].count())
            stats["missing"] = int(df[column].isna().sum())
            stats["dtype"] = str(df[column].dtype)
            
            # Numeric statistics
            if pd.api.types.is_numeric_dtype(df[column]):
                stats["min"] = float(df[column].min())
                stats["max"] = float(df[column].max())
                stats["mean"] = float(df[column].mean())
                stats["median"] = float(df[column].median())
                stats["std"] = float(df[column].std())
                stats["skew"] = float(df[column].skew())
                stats["kurtosis"] = float(df[column].kurtosis())
                
                # Quartiles
                q1 = float(df[column].quantile(0.25))
                q3 = float(df[column].quantile(0.75))
                stats["q1"] = q1
                stats["q3"] = q3
                stats["iqr"] = q3 - q1
                
                # Outliers
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
                stats["outliers_count"] = int(len(outliers))
                stats["outliers_percentage"] = float(len(outliers) / len(df) * 100)
            
            # Categorical statistics
            if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_categorical_dtype(df[column]):
                value_counts = df[column].value_counts()
                stats["unique_values"] = int(len(value_counts))
                stats["top_value"] = str(value_counts.index[0])
                stats["top_value_count"] = int(value_counts.iloc[0])
                stats["top_value_percentage"] = float(value_counts.iloc[0] / len(df) * 100)
                
                # Top 5 values
                top_values = {}
                for i, (value, count) in enumerate(value_counts.items()):
                    if i >= 5:
                        break
                    top_values[str(value)] = int(count)
                stats["top_values"] = top_values
            
            return {
                "success": True,
                "column": column,
                "statistics": stats,
            }
        except Exception as e:
            self.shared_state.add_error(f"Failed to get column statistics: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to get column statistics: {str(e)}",
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
                missing_values[column] = int(df[column].isna().sum())
            self.shared_state.dataset_info.missing_values = missing_values

    async def _process_task(self, task: Task) -> Dict[str, Any]:
        """Process a data cleaning task.
        
        Args:
            task: The task to process.
            
        Returns:
            The result of processing the task.
        """
        if self.shared_state.dataframe is None:
            return {
                "success": False,
                "message": "No dataset has been loaded yet. Cannot clean data.",
            }
        
        # Create a prompt for the model to clean the data
        df = self.shared_state.dataframe
        
        # Get a sample of the data
        sample = df.head(5).to_string()
        
        # Get the dataset info
        dataset_info = self.shared_state.dataset_info.model_dump()
        
        prompt = f"""
        You are a data cleaning agent. Your task is to clean and preprocess the dataset.
        
        Dataset information:
        - Rows: {dataset_info['rows']}
        - Columns: {dataset_info['columns']}
        - Column names: {', '.join(dataset_info['column_names'])}
        - Missing values: {dataset_info['missing_values']}
        
        Here's a sample of the data:
        {sample}
        
        Please analyze the dataset and suggest data cleaning operations. For each operation, explain why it's needed and what it will accomplish.
        
        You have the following tools available:
        - handle_missing_values: Handle missing values in a column
        - remove_duplicates: Remove duplicate rows from the dataset
        - convert_column_type: Convert a column to a different data type
        - normalize_column: Normalize a numeric column
        - drop_columns: Drop columns from the dataset
        - rename_columns: Rename columns in the dataset
        - add_cleaning_insight: Add an insight about the data cleaning process
        - get_column_statistics: Get statistics for a column
        
        For each cleaning operation, call the appropriate tool with the necessary parameters.
        """
        
        # Run the model to generate cleaning operations
        response = await self.run(prompt)
        
        # Return the results
        return {
            "success": True,
            "message": "Data cleaning completed successfully.",
            "model_response": response.get("text", ""),
            "function_calls": response.get("function_call", {}),
            "function_responses": response.get("function_response", {}),
            "dataset_info": self.shared_state.dataset_info.model_dump(),
        }