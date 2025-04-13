"""
Reporting agent for the agent swarm.
"""

from typing import Dict, List, Any, Optional

from shared.state import SharedState, Task, Insight
from agents.base_agent import BaseAgent


class ReportingAgent(BaseAgent):
    """Reporting agent for the agent swarm."""

    def __init__(
        self,
        shared_state: SharedState,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-lite",
    ):
        """Initialize the reporting agent.
        
        Args:
            shared_state: The shared state object.
            api_key: The Google AI Studio API key.
            model_name: The name of the model to use.
        """
        super().__init__(
            name="ReportingAgent",
            description="Compiles insights and visualizations into a coherent report.",
            shared_state=shared_state,
            api_key=api_key,
            model_name=model_name,
        )

    def _register_tools(self) -> None:
        """Register the agent's tools."""
        self.register_tool(self.generate_summary)
        self.register_tool(self.generate_insights_section)
        self.register_tool(self.generate_visualizations_section)
        self.register_tool(self.generate_predictive_modeling_section)
        self.register_tool(self.add_report_insight)

    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of the dataset.
        
        Returns:
            The summary section.
        """
        if self.shared_state.dataframe is None:
            return {
                "success": False,
                "message": "No dataset has been loaded yet.",
            }
        
        try:
            # Get the dataset info
            dataset_info = self.shared_state.dataset_info.model_dump()
            
            # Create the summary
            summary = {
                "title": "Dataset Summary",
                "content": f"""
                This report analyzes the dataset '{self.shared_state.filename}' with {dataset_info['rows']} rows and {dataset_info['columns']} columns.
                
                The dataset contains the following columns:
                {', '.join(dataset_info['column_names'])}
                
                Data types:
                {', '.join([f"{col}: {dtype}" for col, dtype in dataset_info['dtypes'].items()])}
                
                Missing values:
                {', '.join([f"{col}: {count}" for col, count in dataset_info['missing_values'].items() if count > 0])}
                """,
                "dataset_info": dataset_info,
            }
            
            return {
                "success": True,
                "summary": summary,
            }
        except Exception as e:
            self.shared_state.add_error(f"Failed to generate summary: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to generate summary: {str(e)}",
            }

    def generate_insights_section(self) -> Dict[str, Any]:
        """Generate a section about the insights.
        
        Returns:
            The insights section.
        """
        if not self.shared_state.insights:
            return {
                "success": False,
                "message": "No insights available.",
            }
        
        try:
            # Get the insights
            insights = [insight.model_dump() for insight in self.shared_state.insights]
            
            # Sort insights by importance
            insights.sort(key=lambda x: x.get("importance", 0), reverse=True)
            
            # Group insights by category
            insights_by_category = {}
            for insight in insights:
                category = insight.get("category", "general")
                if category not in insights_by_category:
                    insights_by_category[category] = []
                insights_by_category[category].append(insight)
            
            # Create the insights section
            insights_section = {
                "title": "Key Insights",
                "content": "The following insights were discovered during the analysis:\n\n",
                "categories": [],
            }
            
            # Add insights by category
            for category, category_insights in insights_by_category.items():
                category_section = {
                    "category": category,
                    "insights": category_insights,
                }
                insights_section["categories"].append(category_section)
                
                # Add category header
                insights_section["content"] += f"### {category.title()}\n\n"
                
                # Add insights
                for insight in category_insights:
                    insights_section["content"] += f"- {insight['text']}\n"
                
                insights_section["content"] += "\n"
            
            return {
                "success": True,
                "insights_section": insights_section,
            }
        except Exception as e:
            self.shared_state.add_error(f"Failed to generate insights section: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to generate insights section: {str(e)}",
            }

    def generate_visualizations_section(self) -> Dict[str, Any]:
        """Generate a section about the visualizations.
        
        Returns:
            The visualizations section.
        """
        if not self.shared_state.visualizations:
            return {
                "success": False,
                "message": "No visualizations available.",
            }
        
        try:
            # Get the visualizations
            visualizations = [viz.model_dump() for viz in self.shared_state.visualizations]
            
            # Group visualizations by type
            visualizations_by_type = {}
            for viz in visualizations:
                viz_type = viz.get("type", "general")
                if viz_type not in visualizations_by_type:
                    visualizations_by_type[viz_type] = []
                visualizations_by_type[viz_type].append(viz)
            
            # Create the visualizations section
            visualizations_section = {
                "title": "Visualizations",
                "content": "The following visualizations were created during the analysis:\n\n",
                "types": [],
            }
            
            # Add visualizations by type
            for viz_type, type_visualizations in visualizations_by_type.items():
                type_section = {
                    "type": viz_type,
                    "visualizations": type_visualizations,
                }
                visualizations_section["types"].append(type_section)
                
                # Add type header
                visualizations_section["content"] += f"### {viz_type.replace('_', ' ').title()}\n\n"
                
                # Add visualizations
                for viz in type_visualizations:
                    visualizations_section["content"] += f"#### {viz['title']}\n\n"
                    visualizations_section["content"] += f"{viz['description']}\n\n"
                    # Note: In a real report, we would include the visualization image here
                    visualizations_section["content"] += f"[Visualization: {viz['title']}]\n\n"
            
            return {
                "success": True,
                "visualizations_section": visualizations_section,
            }
        except Exception as e:
            self.shared_state.add_error(f"Failed to generate visualizations section: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to generate visualizations section: {str(e)}",
            }

    def generate_predictive_modeling_section(self) -> Dict[str, Any]:
        """Generate a section about the predictive modeling.
        
        Returns:
            The predictive modeling section.
        """
        if not self.shared_state.predictive_models:
            return {
                "success": False,
                "message": "No predictive models available.",
            }
        
        try:
            # Get the predictive models
            models = [model.model_dump() for model in self.shared_state.predictive_models]
            
            # Create the predictive modeling section
            modeling_section = {
                "title": "Predictive Modeling",
                "content": "The following predictive models were created during the analysis:\n\n",
                "models": models,
            }
            
            # Add models
            for model in models:
                model_type = model.get("model_type", "unknown")
                target_column = model.get("target_column", "unknown")
                
                modeling_section["content"] += f"### {model_type.title()} Model for {target_column}\n\n"
                
                # Add metrics
                metrics = model.get("metrics", {})
                modeling_section["content"] += "#### Metrics\n\n"
                for metric, value in metrics.items():
                    modeling_section["content"] += f"- {metric}: {value}\n"
                
                modeling_section["content"] += "\n"
                
                # Add feature importance
                feature_importance = model.get("feature_importance", {})
                top_features = model.get("top_features", [])
                
                modeling_section["content"] += "#### Feature Importance\n\n"
                for feature in top_features:
                    importance = feature_importance.get(feature, 0)
                    modeling_section["content"] += f"- {feature}: {importance}\n"
                
                modeling_section["content"] += "\n"
            
            return {
                "success": True,
                "modeling_section": modeling_section,
            }
        except Exception as e:
            self.shared_state.add_error(f"Failed to generate predictive modeling section: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to generate predictive modeling section: {str(e)}",
            }

    def add_report_insight(self, text: str, importance: int = 1, category: str = "") -> Dict[str, Any]:
        """Add an insight about the report.
        
        Args:
            text: The text of the insight.
            importance: The importance of the insight (1-5).
            category: The category of the insight.
            
        Returns:
            Information about the added insight.
        """
        insight = Insight(
            text=text,
            importance=importance,
            category=category,
            related_columns=[],
        )
        
        self.shared_state.add_insight(insight)
        
        return {
            "success": True,
            "message": "Insight added successfully.",
            "insight": insight.model_dump(),
        }

    async def _process_task(self, task: Task) -> Dict[str, Any]:
        """Process a reporting task.
        
        Args:
            task: The task to process.
            
        Returns:
            The result of processing the task.
        """
        if self.shared_state.dataframe is None:
            return {
                "success": False,
                "message": "No dataset has been loaded yet. Cannot generate report.",
            }
        
        # Create a prompt for the model to generate a report
        df = self.shared_state.dataframe
        
        # Get the dataset info
        dataset_info = self.shared_state.dataset_info.model_dump()
        
        # Get the insights
        insights = [insight.model_dump() for insight in self.shared_state.insights]
        
        # Get the visualizations
        visualizations = [viz.model_dump() for viz in self.shared_state.visualizations]
        
        # Get the predictive models
        models = [model.model_dump() for model in self.shared_state.predictive_models]
        
        prompt = f"""
        You are a reporting agent. Your task is to compile insights and visualizations into a coherent report.
        
        Dataset information:
        - Filename: {self.shared_state.filename}
        - Rows: {dataset_info['rows']}
        - Columns: {dataset_info['columns']}
        - Column names: {', '.join(dataset_info['column_names'])}
        
        Please generate a comprehensive report that includes:
        1. A summary of the dataset
        2. Key insights from the analysis
        3. Visualizations and their interpretations
        4. Predictive modeling results (if available)
        5. Recommendations based on the analysis
        
        You have the following tools available:
        - generate_summary: Generate a summary of the dataset
        - generate_insights_section: Generate a section about the insights
        - generate_visualizations_section: Generate a section about the visualizations
        - generate_predictive_modeling_section: Generate a section about the predictive modeling
        - add_report_insight: Add an insight about the report
        
        Call these tools to generate the different sections of the report.
        """
        
        # Run the model to generate the report
        response = await self.run(prompt)
        
        # Compile the report
        report = {
            "title": f"Data Analysis Report: {self.shared_state.filename}",
            "sections": [],
        }
        
        # Add the summary section
        summary_result = self.generate_summary()
        if summary_result.get("success", False):
            report["sections"].append(summary_result["summary"])
        
        # Add the insights section
        insights_result = self.generate_insights_section()
        if insights_result.get("success", False):
            report["sections"].append(insights_result["insights_section"])
        
        # Add the visualizations section
        visualizations_result = self.generate_visualizations_section()
        if visualizations_result.get("success", False):
            report["sections"].append(visualizations_result["visualizations_section"])
        
        # Add the predictive modeling section
        modeling_result = self.generate_predictive_modeling_section()
        if modeling_result.get("success", False):
            report["sections"].append(modeling_result["modeling_section"])
        
        # Return the results
        return {
            "success": True,
            "message": "Report generated successfully.",
            "model_response": response.get("text", ""),
            "function_calls": response.get("function_call", {}),
            "function_responses": response.get("function_response", {}),
            "report": report,
        }