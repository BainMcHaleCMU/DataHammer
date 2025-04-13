"""
Shared state module for the agent swarm.
This module provides a central place for agents to store and retrieve data.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
from pydantic import BaseModel, Field
from enum import Enum


class DatasetInfo(BaseModel):
    """Information about the dataset."""
    rows: int = 0
    columns: int = 0
    column_names: List[str] = Field(default_factory=list)
    dtypes: Dict[str, str] = Field(default_factory=dict)
    missing_values: Dict[str, int] = Field(default_factory=dict)


class Visualization(BaseModel):
    """Visualization data."""
    title: str
    description: str
    type: str  # e.g., "heatmap", "histogram", "scatter", etc.
    data: str  # Base64 encoded image
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Insight(BaseModel):
    """Data insight."""
    text: str
    importance: int = 1  # 1-5 scale
    category: str = ""  # e.g., "correlation", "outlier", "trend", etc.
    related_columns: List[str] = Field(default_factory=list)


class PredictiveModel(BaseModel):
    """Information about a predictive model."""
    model_type: str  # e.g., "regression", "classification"
    target_column: str
    feature_importance: Dict[str, float] = Field(default_factory=dict)
    top_features: List[str] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)


class TaskStatus(str, Enum):
    """Status of a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(str, Enum):
    """Type of task."""
    DATA_LOADING = "data_loading"
    DATA_CLEANING = "data_cleaning"
    DATA_ANALYSIS = "data_analysis"
    DATA_VISUALIZATION = "data_visualization"
    REPORTING = "reporting"


class Task(BaseModel):
    """Task for an agent to perform."""
    task_id: str
    task_type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    description: str
    assigned_to: str = ""
    created_at: str = ""
    completed_at: Optional[str] = None
    result: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class SharedState:
    """Shared state for the agent swarm."""

    def __init__(self):
        """Initialize the shared state."""
        self.dataframe: Optional[pd.DataFrame] = None
        self.dataset_info: DatasetInfo = DatasetInfo()
        self.insights: List[Insight] = []
        self.visualizations: List[Visualization] = []
        self.predictive_models: List[PredictiveModel] = []
        self.tasks: List[Task] = []
        self.task_history: List[Task] = []
        self.metadata: Dict[str, Any] = {}
        self.filename: str = ""
        self.file_extension: str = ""
        self.file_size: int = 0
        self.processing_stage: str = ""
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert the shared state to a dictionary."""
        return {
            "dataset_info": self.dataset_info.model_dump() if self.dataset_info else {},
            "insights": [insight.model_dump() for insight in self.insights],
            "visualizations": [viz.model_dump() for viz in self.visualizations],
            "predictive_modeling": {
                "models": [model.model_dump() for model in self.predictive_models]
            } if self.predictive_models else {},
            "metadata": {
                "filename": self.filename,
                "file_extension": self.file_extension,
                "file_size": self.file_size,
                "processing_stage": self.processing_stage,
            },
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def add_task(self, task: Task) -> None:
        """Add a task to the task list."""
        self.tasks.append(task)

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def update_task(self, task_id: str, **kwargs) -> None:
        """Update a task."""
        for i, task in enumerate(self.tasks):
            if task.task_id == task_id:
                for key, value in kwargs.items():
                    setattr(task, key, value)
                break

    def complete_task(self, task_id: str, result: Dict[str, Any]) -> None:
        """Mark a task as completed."""
        from datetime import datetime
        for i, task in enumerate(self.tasks):
            if task.task_id == task_id:
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now().isoformat()
                task.result = result
                self.task_history.append(task)
                self.tasks.pop(i)
                break

    def fail_task(self, task_id: str, error: str) -> None:
        """Mark a task as failed."""
        from datetime import datetime
        for i, task in enumerate(self.tasks):
            if task.task_id == task_id:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now().isoformat()
                task.error = error
                self.task_history.append(task)
                self.tasks.pop(i)
                break

    def get_pending_tasks(self, task_type: Optional[TaskType] = None) -> List[Task]:
        """Get pending tasks."""
        if task_type:
            return [task for task in self.tasks if task.status == TaskStatus.PENDING and task.task_type == task_type]
        return [task for task in self.tasks if task.status == TaskStatus.PENDING]

    def add_insight(self, insight: Insight) -> None:
        """Add an insight to the shared state."""
        self.insights.append(insight)

    def add_visualization(self, visualization: Visualization) -> None:
        """Add a visualization to the shared state."""
        self.visualizations.append(visualization)

    def add_predictive_model(self, model: PredictiveModel) -> None:
        """Add a predictive model to the shared state."""
        self.predictive_models.append(model)

    def add_error(self, error: str) -> None:
        """Add an error to the shared state."""
        self.errors.append(error)

    def add_warning(self, warning: str) -> None:
        """Add a warning to the shared state."""
        self.warnings.append(warning)


# Create a global shared state instance
shared_state = SharedState()