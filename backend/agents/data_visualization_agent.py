"""
Data visualization agent for the agent swarm.
"""

import io
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Union

from shared.state import SharedState, Task, Insight, Visualization
from agents.base_agent import BaseAgent


class DataVisualizationAgent(BaseAgent):
    """Data visualization agent for the agent swarm."""

    def __init__(
        self,
        shared_state: SharedState,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-lite",
    ):
        """Initialize the data visualization agent.
        
        Args:
            shared_state: The shared state object.
            api_key: The Google AI Studio API key.
            model_name: The name of the model to use.
        """
        super().__init__(
            name="DataVisualizationAgent",
            description="Creates visualizations based on data.",
            shared_state=shared_state,
            api_key=api_key,
            model_name=model_name,
        )

    def _register_tools(self) -> None:
        """Register the agent's tools."""
        self.register_tool(self.create_histogram)
        self.register_tool(self.create_bar_chart)
        self.register_tool(self.create_scatter_plot)
        self.register_tool(self.create_line_chart)
        self.register_tool(self.create_heatmap)
        self.register_tool(self.create_box_plot)
        self.register_tool(self.create_pie_chart)
        self.register_tool(self.add_visualization_insight)

    def _fig_to_base64(self, fig) -> str:
        """Convert a matplotlib figure to a base64 encoded string.
        
        Args:
            fig: The matplotlib figure.
            
        Returns:
            The base64 encoded string.
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)
        return img_str

    def _plotly_to_base64(self, fig) -> str:
        """Convert a plotly figure to a base64 encoded string.
        
        Args:
            fig: The plotly figure.
            
        Returns:
            The base64 encoded string.
        """
        img_bytes = fig.to_image(format="png", width=800, height=600, scale=1)
        img_str = base64.b64encode(img_bytes).decode('utf-8')
        return img_str

    def create_histogram(
        self,
        column: str,
        bins: int = 10,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a histogram for a numeric column.
        
        Args:
            column: The name of the column.
            bins: The number of bins.
            title: The title of the histogram.
            xlabel: The label for the x-axis.
            ylabel: The label for the y-axis.
            description: A description of the visualization.
            
        Returns:
            The base64 encoded histogram.
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
                "message": f"Column '{column}' is not numeric.",
            }
        
        try:
            # Create the histogram using plotly
            fig = px.histogram(
                df,
                x=column,
                nbins=bins,
                title=title or f"Histogram of {column}",
                labels={column: xlabel or column},
            )
            
            # Update the layout
            fig.update_layout(
                xaxis_title=xlabel or column,
                yaxis_title=ylabel or "Count",
                template="plotly_white",
            )
            
            # Convert to base64
            img_str = self._plotly_to_base64(fig)
            
            # Create a visualization object
            visualization = Visualization(
                title=title or f"Histogram of {column}",
                description=description or f"Distribution of values in column '{column}'.",
                type="histogram",
                data=img_str,
                metadata={
                    "column": column,
                    "bins": bins,
                    "min": float(df[column].min()),
                    "max": float(df[column].max()),
                    "mean": float(df[column].mean()),
                    "median": float(df[column].median()),
                    "std": float(df[column].std()),
                },
            )
            
            # Add the visualization to the shared state
            self.shared_state.add_visualization(visualization)
            
            # Add an insight about the visualization
            self.add_visualization_insight(
                text=f"Created histogram showing distribution of '{column}'.",
                importance=3,
                category="visualization",
                related_columns=[column],
            )
            
            return {
                "success": True,
                "message": f"Successfully created histogram for column '{column}'.",
                "visualization": visualization.model_dump(),
            }
        except Exception as e:
            self.shared_state.add_error(f"Failed to create histogram: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to create histogram: {str(e)}",
            }

    def create_bar_chart(
        self,
        x_column: str,
        y_column: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        description: Optional[str] = None,
        horizontal: bool = False,
        top_n: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create a bar chart.
        
        Args:
            x_column: The name of the column for the x-axis.
            y_column: The name of the column for the y-axis. If None, counts are used.
            title: The title of the bar chart.
            xlabel: The label for the x-axis.
            ylabel: The label for the y-axis.
            description: A description of the visualization.
            horizontal: Whether to create a horizontal bar chart.
            top_n: The number of top values to include. If None, all values are included.
            
        Returns:
            The base64 encoded bar chart.
        """
        if self.shared_state.dataframe is None:
            return {
                "success": False,
                "message": "No dataset has been loaded yet.",
            }
        
        df = self.shared_state.dataframe
        
        if x_column not in df.columns:
            return {
                "success": False,
                "message": f"Column '{x_column}' not found in the dataset.",
            }
        
        if y_column is not None and y_column not in df.columns:
            return {
                "success": False,
                "message": f"Column '{y_column}' not found in the dataset.",
            }
        
        try:
            # Prepare the data
            if y_column is None:
                # Count the occurrences of each value in x_column
                data = df[x_column].value_counts().reset_index()
                data.columns = [x_column, 'count']
                x = x_column
                y = 'count'
            else:
                # Use the provided y_column
                data = df
                x = x_column
                y = y_column
            
            # Limit to top N values if specified
            if top_n is not None and y_column is None:
                data = data.head(top_n)
            
            # Create the bar chart using plotly
            if horizontal:
                fig = px.bar(
                    data,
                    y=x,
                    x=y,
                    title=title or f"Bar Chart of {x} vs {y}",
                    labels={x: xlabel or x, y: ylabel or y},
                    orientation='h',
                )
            else:
                fig = px.bar(
                    data,
                    x=x,
                    y=y,
                    title=title or f"Bar Chart of {x} vs {y}",
                    labels={x: xlabel or x, y: ylabel or y},
                )
            
            # Update the layout
            fig.update_layout(
                xaxis_title=xlabel or x,
                yaxis_title=ylabel or y,
                template="plotly_white",
            )
            
            # Convert to base64
            img_str = self._plotly_to_base64(fig)
            
            # Create a visualization object
            visualization = Visualization(
                title=title or f"Bar Chart of {x} vs {y}",
                description=description or f"Bar chart showing relationship between '{x}' and '{y}'.",
                type="bar_chart",
                data=img_str,
                metadata={
                    "x_column": x,
                    "y_column": y,
                    "horizontal": horizontal,
                    "top_n": top_n,
                },
            )
            
            # Add the visualization to the shared state
            self.shared_state.add_visualization(visualization)
            
            # Add an insight about the visualization
            self.add_visualization_insight(
                text=f"Created bar chart showing relationship between '{x}' and '{y}'.",
                importance=3,
                category="visualization",
                related_columns=[x_column, y_column] if y_column else [x_column],
            )
            
            return {
                "success": True,
                "message": f"Successfully created bar chart for columns '{x}' and '{y}'.",
                "visualization": visualization.model_dump(),
            }
        except Exception as e:
            self.shared_state.add_error(f"Failed to create bar chart: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to create bar chart: {str(e)}",
            }

    def create_scatter_plot(
        self,
        x_column: str,
        y_column: str,
        color_column: Optional[str] = None,
        size_column: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        description: Optional[str] = None,
        add_trendline: bool = False,
    ) -> Dict[str, Any]:
        """Create a scatter plot.
        
        Args:
            x_column: The name of the column for the x-axis.
            y_column: The name of the column for the y-axis.
            color_column: The name of the column for the color.
            size_column: The name of the column for the size.
            title: The title of the scatter plot.
            xlabel: The label for the x-axis.
            ylabel: The label for the y-axis.
            description: A description of the visualization.
            add_trendline: Whether to add a trendline.
            
        Returns:
            The base64 encoded scatter plot.
        """
        if self.shared_state.dataframe is None:
            return {
                "success": False,
                "message": "No dataset has been loaded yet.",
            }
        
        df = self.shared_state.dataframe
        
        if x_column not in df.columns:
            return {
                "success": False,
                "message": f"Column '{x_column}' not found in the dataset.",
            }
        
        if y_column not in df.columns:
            return {
                "success": False,
                "message": f"Column '{y_column}' not found in the dataset.",
            }
        
        if color_column is not None and color_column not in df.columns:
            return {
                "success": False,
                "message": f"Column '{color_column}' not found in the dataset.",
            }
        
        if size_column is not None and size_column not in df.columns:
            return {
                "success": False,
                "message": f"Column '{size_column}' not found in the dataset.",
            }
        
        if not pd.api.types.is_numeric_dtype(df[x_column]):
            return {
                "success": False,
                "message": f"Column '{x_column}' is not numeric.",
            }
        
        if not pd.api.types.is_numeric_dtype(df[y_column]):
            return {
                "success": False,
                "message": f"Column '{y_column}' is not numeric.",
            }
        
        if size_column is not None and not pd.api.types.is_numeric_dtype(df[size_column]):
            return {
                "success": False,
                "message": f"Column '{size_column}' is not numeric.",
            }
        
        try:
            # Create the scatter plot using plotly
            fig = px.scatter(
                df,
                x=x_column,
                y=y_column,
                color=color_column,
                size=size_column,
                title=title or f"Scatter Plot of {x_column} vs {y_column}",
                labels={
                    x_column: xlabel or x_column,
                    y_column: ylabel or y_column,
                    color_column: color_column,
                    size_column: size_column,
                },
                trendline="ols" if add_trendline else None,
            )
            
            # Update the layout
            fig.update_layout(
                xaxis_title=xlabel or x_column,
                yaxis_title=ylabel or y_column,
                template="plotly_white",
            )
            
            # Convert to base64
            img_str = self._plotly_to_base64(fig)
            
            # Calculate correlation
            correlation = df[[x_column, y_column]].corr().loc[x_column, y_column]
            
            # Create a visualization object
            visualization = Visualization(
                title=title or f"Scatter Plot of {x_column} vs {y_column}",
                description=description or f"Scatter plot showing relationship between '{x_column}' and '{y_column}'.",
                type="scatter_plot",
                data=img_str,
                metadata={
                    "x_column": x_column,
                    "y_column": y_column,
                    "color_column": color_column,
                    "size_column": size_column,
                    "correlation": float(correlation),
                    "add_trendline": add_trendline,
                },
            )
            
            # Add the visualization to the shared state
            self.shared_state.add_visualization(visualization)
            
            # Add an insight about the visualization
            self.add_visualization_insight(
                text=f"Created scatter plot showing relationship between '{x_column}' and '{y_column}' with correlation of {correlation:.2f}.",
                importance=3,
                category="visualization",
                related_columns=[x_column, y_column] + ([color_column] if color_column else []) + ([size_column] if size_column else []),
            )
            
            return {
                "success": True,
                "message": f"Successfully created scatter plot for columns '{x_column}' and '{y_column}'.",
                "visualization": visualization.model_dump(),
            }
        except Exception as e:
            self.shared_state.add_error(f"Failed to create scatter plot: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to create scatter plot: {str(e)}",
            }

    def create_line_chart(
        self,
        x_column: str,
        y_columns: List[str],
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a line chart.
        
        Args:
            x_column: The name of the column for the x-axis.
            y_columns: The names of the columns for the y-axis.
            title: The title of the line chart.
            xlabel: The label for the x-axis.
            ylabel: The label for the y-axis.
            description: A description of the visualization.
            
        Returns:
            The base64 encoded line chart.
        """
        if self.shared_state.dataframe is None:
            return {
                "success": False,
                "message": "No dataset has been loaded yet.",
            }
        
        df = self.shared_state.dataframe
        
        if x_column not in df.columns:
            return {
                "success": False,
                "message": f"Column '{x_column}' not found in the dataset.",
            }
        
        for y_column in y_columns:
            if y_column not in df.columns:
                return {
                    "success": False,
                    "message": f"Column '{y_column}' not found in the dataset.",
                }
            
            if not pd.api.types.is_numeric_dtype(df[y_column]):
                return {
                    "success": False,
                    "message": f"Column '{y_column}' is not numeric.",
                }
        
        try:
            # Create the line chart using plotly
            fig = go.Figure()
            
            for y_column in y_columns:
                fig.add_trace(
                    go.Scatter(
                        x=df[x_column],
                        y=df[y_column],
                        mode='lines+markers',
                        name=y_column,
                    )
                )
            
            # Update the layout
            fig.update_layout(
                title=title or f"Line Chart of {', '.join(y_columns)} vs {x_column}",
                xaxis_title=xlabel or x_column,
                yaxis_title=ylabel or "Value",
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            
            # Convert to base64
            img_str = self._plotly_to_base64(fig)
            
            # Create a visualization object
            visualization = Visualization(
                title=title or f"Line Chart of {', '.join(y_columns)} vs {x_column}",
                description=description or f"Line chart showing trends of {', '.join(y_columns)} over {x_column}.",
                type="line_chart",
                data=img_str,
                metadata={
                    "x_column": x_column,
                    "y_columns": y_columns,
                },
            )
            
            # Add the visualization to the shared state
            self.shared_state.add_visualization(visualization)
            
            # Add an insight about the visualization
            self.add_visualization_insight(
                text=f"Created line chart showing trends of {', '.join(y_columns)} over {x_column}.",
                importance=3,
                category="visualization",
                related_columns=[x_column] + y_columns,
            )
            
            return {
                "success": True,
                "message": f"Successfully created line chart for columns '{x_column}' and '{', '.join(y_columns)}'.",
                "visualization": visualization.model_dump(),
            }
        except Exception as e:
            self.shared_state.add_error(f"Failed to create line chart: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to create line chart: {str(e)}",
            }

    def create_heatmap(
        self,
        columns: Optional[List[str]] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a correlation heatmap.
        
        Args:
            columns: The columns to include in the heatmap. If None, all numeric columns are used.
            title: The title of the heatmap.
            description: A description of the visualization.
            
        Returns:
            The base64 encoded heatmap.
        """
        if self.shared_state.dataframe is None:
            return {
                "success": False,
                "message": "No dataset has been loaded yet.",
            }
        
        df = self.shared_state.dataframe
        
        # Filter numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_columns:
            return {
                "success": False,
                "message": "No numeric columns found in the dataset.",
            }
        
        if columns:
            # Check if all specified columns exist and are numeric
            for column in columns:
                if column not in df.columns:
                    return {
                        "success": False,
                        "message": f"Column '{column}' not found in the dataset.",
                    }
                if column not in numeric_columns:
                    return {
                        "success": False,
                        "message": f"Column '{column}' is not numeric.",
                    }
            
            # Use only the specified columns
            numeric_columns = columns
        
        try:
            # Calculate correlation
            correlation = df[numeric_columns].corr().round(2)
            
            # Create the heatmap using plotly
            fig = px.imshow(
                correlation,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title=title or "Correlation Heatmap",
            )
            
            # Update the layout
            fig.update_layout(
                template="plotly_white",
            )
            
            # Convert to base64
            img_str = self._plotly_to_base64(fig)
            
            # Find strong correlations
            strong_correlations = []
            for i, col1 in enumerate(numeric_columns):
                for j, col2 in enumerate(numeric_columns):
                    if i < j:  # Only consider each pair once
                        corr_value = correlation.loc[col1, col2]
                        if abs(corr_value) >= 0.7:  # Strong correlation threshold
                            strong_correlations.append({
                                "column1": col1,
                                "column2": col2,
                                "correlation": float(corr_value),
                                "type": "positive" if corr_value > 0 else "negative",
                            })
            
            # Create a visualization object
            visualization = Visualization(
                title=title or "Correlation Heatmap",
                description=description or "Heatmap showing correlation between numeric columns.",
                type="heatmap",
                data=img_str,
                metadata={
                    "columns": numeric_columns,
                    "strong_correlations": strong_correlations,
                },
            )
            
            # Add the visualization to the shared state
            self.shared_state.add_visualization(visualization)
            
            # Add an insight about the visualization
            self.add_visualization_insight(
                text=f"Created correlation heatmap for {len(numeric_columns)} numeric columns.",
                importance=3,
                category="visualization",
                related_columns=numeric_columns,
            )
            
            return {
                "success": True,
                "message": f"Successfully created correlation heatmap.",
                "visualization": visualization.model_dump(),
            }
        except Exception as e:
            self.shared_state.add_error(f"Failed to create heatmap: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to create heatmap: {str(e)}",
            }

    def create_box_plot(
        self,
        column: str,
        group_by: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a box plot.
        
        Args:
            column: The name of the column to plot.
            group_by: The name of the column to group by.
            title: The title of the box plot.
            xlabel: The label for the x-axis.
            ylabel: The label for the y-axis.
            description: A description of the visualization.
            
        Returns:
            The base64 encoded box plot.
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
                "message": f"Column '{column}' is not numeric.",
            }
        
        if group_by is not None and group_by not in df.columns:
            return {
                "success": False,
                "message": f"Column '{group_by}' not found in the dataset.",
            }
        
        try:
            # Create the box plot using plotly
            fig = px.box(
                df,
                y=column,
                x=group_by,
                title=title or f"Box Plot of {column}" + (f" grouped by {group_by}" if group_by else ""),
                labels={
                    column: ylabel or column,
                    group_by: xlabel or group_by,
                },
            )
            
            # Update the layout
            fig.update_layout(
                xaxis_title=xlabel or group_by,
                yaxis_title=ylabel or column,
                template="plotly_white",
            )
            
            # Convert to base64
            img_str = self._plotly_to_base64(fig)
            
            # Calculate statistics
            stats = df[column].describe().to_dict()
            
            # Create a visualization object
            visualization = Visualization(
                title=title or f"Box Plot of {column}" + (f" grouped by {group_by}" if group_by else ""),
                description=description or f"Box plot showing distribution of '{column}'" + (f" grouped by '{group_by}'" if group_by else "."),
                type="box_plot",
                data=img_str,
                metadata={
                    "column": column,
                    "group_by": group_by,
                    "statistics": {k: float(v) for k, v in stats.items()},
                },
            )
            
            # Add the visualization to the shared state
            self.shared_state.add_visualization(visualization)
            
            # Add an insight about the visualization
            self.add_visualization_insight(
                text=f"Created box plot showing distribution of '{column}'" + (f" grouped by '{group_by}'" if group_by else "."),
                importance=3,
                category="visualization",
                related_columns=[column] + ([group_by] if group_by else []),
            )
            
            return {
                "success": True,
                "message": f"Successfully created box plot for column '{column}'.",
                "visualization": visualization.model_dump(),
            }
        except Exception as e:
            self.shared_state.add_error(f"Failed to create box plot: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to create box plot: {str(e)}",
            }

    def create_pie_chart(
        self,
        column: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create a pie chart.
        
        Args:
            column: The name of the column to plot.
            title: The title of the pie chart.
            description: A description of the visualization.
            top_n: The number of top values to include. If None, all values are included.
            
        Returns:
            The base64 encoded pie chart.
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
            # Count the occurrences of each value in the column
            value_counts = df[column].value_counts()
            
            # Limit to top N values if specified
            if top_n is not None and len(value_counts) > top_n:
                # Get the top N values
                top_values = value_counts.head(top_n)
                
                # Add an "Other" category for the rest
                other_count = value_counts.iloc[top_n:].sum()
                top_values = pd.concat([top_values, pd.Series([other_count], index=["Other"])])
                
                # Use the top values
                value_counts = top_values
            
            # Create the pie chart using plotly
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=title or f"Pie Chart of {column}",
            )
            
            # Update the layout
            fig.update_layout(
                template="plotly_white",
            )
            
            # Convert to base64
            img_str = self._plotly_to_base64(fig)
            
            # Create a visualization object
            visualization = Visualization(
                title=title or f"Pie Chart of {column}",
                description=description or f"Pie chart showing distribution of values in column '{column}'.",
                type="pie_chart",
                data=img_str,
                metadata={
                    "column": column,
                    "top_n": top_n,
                    "value_counts": {str(k): int(v) for k, v in value_counts.items()},
                },
            )
            
            # Add the visualization to the shared state
            self.shared_state.add_visualization(visualization)
            
            # Add an insight about the visualization
            self.add_visualization_insight(
                text=f"Created pie chart showing distribution of values in column '{column}'.",
                importance=3,
                category="visualization",
                related_columns=[column],
            )
            
            return {
                "success": True,
                "message": f"Successfully created pie chart for column '{column}'.",
                "visualization": visualization.model_dump(),
            }
        except Exception as e:
            self.shared_state.add_error(f"Failed to create pie chart: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to create pie chart: {str(e)}",
            }

    def add_visualization_insight(self, text: str, importance: int = 1, category: str = "", related_columns: List[str] = None) -> Dict[str, Any]:
        """Add an insight about a visualization.
        
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
        """Process a data visualization task.
        
        Args:
            task: The task to process.
            
        Returns:
            The result of processing the task.
        """
        if self.shared_state.dataframe is None:
            return {
                "success": False,
                "message": "No dataset has been loaded yet. Cannot create visualizations.",
            }
        
        # Create a prompt for the model to create visualizations
        df = self.shared_state.dataframe
        
        # Get a sample of the data
        sample = df.head(5).to_string()
        
        # Get the dataset info
        dataset_info = self.shared_state.dataset_info.model_dump()
        
        # Get the insights
        insights = [insight.model_dump() for insight in self.shared_state.insights]
        
        prompt = f"""
        You are a data visualization agent. Your task is to create visualizations based on the dataset.
        
        Dataset information:
        - Rows: {dataset_info['rows']}
        - Columns: {dataset_info['columns']}
        - Column names: {', '.join(dataset_info['column_names'])}
        - Missing values: {dataset_info['missing_values']}
        
        Here's a sample of the data:
        {sample}
        
        Insights from data analysis:
        {insights}
        
        Please create visualizations that help understand the data. For each visualization, explain what it shows and why it's important.
        
        You have the following tools available:
        - create_histogram: Create a histogram for a numeric column
        - create_bar_chart: Create a bar chart
        - create_scatter_plot: Create a scatter plot
        - create_line_chart: Create a line chart
        - create_heatmap: Create a correlation heatmap
        - create_box_plot: Create a box plot
        - create_pie_chart: Create a pie chart
        - add_visualization_insight: Add an insight about a visualization
        
        For each visualization, call the appropriate tool with the necessary parameters.
        """
        
        # Run the model to generate visualizations
        response = await self.run(prompt)
        
        # Return the results
        return {
            "success": True,
            "message": "Data visualization completed successfully.",
            "model_response": response.get("text", ""),
            "function_calls": response.get("function_call", {}),
            "function_responses": response.get("function_response", {}),
            "visualizations": [viz.model_dump() for viz in self.shared_state.visualizations],
        }