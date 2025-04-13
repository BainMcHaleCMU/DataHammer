"""
Data analysis agent for the agent swarm.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

from shared.state import SharedState, Task, Insight, PredictiveModel
from agents.base_agent import BaseAgent


class DataAnalysisAgent(BaseAgent):
    """Data analysis agent for the agent swarm."""

    def __init__(
        self,
        shared_state: SharedState,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-lite",
    ):
        """Initialize the data analysis agent.
        
        Args:
            shared_state: The shared state object.
            api_key: The Google AI Studio API key.
            model_name: The name of the model to use.
        """
        super().__init__(
            name="DataAnalysisAgent",
            description="Analyzes data and generates insights.",
            shared_state=shared_state,
            api_key=api_key,
            model_name=model_name,
        )

    def _register_tools(self) -> None:
        """Register the agent's tools."""
        self.register_tool(self.calculate_correlation)
        self.register_tool(self.calculate_summary_statistics)
        self.register_tool(self.detect_outliers)
        self.register_tool(self.perform_regression)
        self.register_tool(self.perform_classification)
        self.register_tool(self.add_analysis_insight)
        self.register_tool(self.get_column_distribution)

    def calculate_correlation(self, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculate correlation between numeric columns.
        
        Args:
            columns: The columns to include in the correlation calculation. If None, all numeric columns are used.
            
        Returns:
            The correlation matrix.
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
            correlation = df[numeric_columns].corr().fillna(0).round(3)
            
            # Convert to dictionary
            correlation_dict = {}
            for column in correlation.columns:
                correlation_dict[column] = {}
                for index in correlation.index:
                    correlation_dict[column][index] = float(correlation.loc[index, column])
            
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
            
            # Add insights about strong correlations
            for corr in strong_correlations:
                self.add_analysis_insight(
                    text=f"Strong {corr['type']} correlation ({corr['correlation']:.2f}) between '{corr['column1']}' and '{corr['column2']}'.",
                    importance=4,
                    category="correlation",
                    related_columns=[corr['column1'], corr['column2']],
                )
            
            return {
                "success": True,
                "correlation": correlation_dict,
                "strong_correlations": strong_correlations,
            }
        except Exception as e:
            self.shared_state.add_error(f"Failed to calculate correlation: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to calculate correlation: {str(e)}",
            }

    def calculate_summary_statistics(self, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculate summary statistics for numeric columns.
        
        Args:
            columns: The columns to include in the calculation. If None, all numeric columns are used.
            
        Returns:
            Summary statistics for the columns.
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
            # Calculate summary statistics
            summary = df[numeric_columns].describe().round(3)
            
            # Convert to dictionary
            summary_dict = {}
            for column in summary.columns:
                summary_dict[column] = {}
                for index in summary.index:
                    summary_dict[column][index] = float(summary.loc[index, column])
                
                # Add skewness and kurtosis
                summary_dict[column]["skew"] = float(df[column].skew())
                summary_dict[column]["kurtosis"] = float(df[column].kurtosis())
            
            # Add insights about skewed distributions
            for column in numeric_columns:
                skew = df[column].skew()
                if abs(skew) >= 1.0:
                    skew_type = "positively" if skew > 0 else "negatively"
                    self.add_analysis_insight(
                        text=f"Column '{column}' is {skew_type} skewed ({skew:.2f}).",
                        importance=3,
                        category="distribution",
                        related_columns=[column],
                    )
            
            return {
                "success": True,
                "summary_statistics": summary_dict,
            }
        except Exception as e:
            self.shared_state.add_error(f"Failed to calculate summary statistics: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to calculate summary statistics: {str(e)}",
            }

    def detect_outliers(self, column: str, method: str = 'iqr') -> Dict[str, Any]:
        """Detect outliers in a numeric column.
        
        Args:
            column: The name of the column.
            method: The method to use for outlier detection ('iqr', 'zscore').
            
        Returns:
            Information about the outliers.
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
            outliers = []
            
            if method == 'iqr':
                # IQR method
                q1 = df[column].quantile(0.25)
                q3 = df[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Find outliers
                outlier_indices = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index.tolist()
                outlier_values = df.loc[outlier_indices, column].tolist()
                
                # Create outlier information
                for i, idx in enumerate(outlier_indices):
                    outliers.append({
                        "index": int(idx),
                        "value": float(outlier_values[i]),
                        "type": "low" if outlier_values[i] < lower_bound else "high",
                    })
                
                # Add insight about outliers
                if outliers:
                    self.add_analysis_insight(
                        text=f"Found {len(outliers)} outliers in column '{column}' using IQR method.",
                        importance=3,
                        category="outliers",
                        related_columns=[column],
                    )
                
                return {
                    "success": True,
                    "method": "iqr",
                    "column": column,
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(iqr),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "outliers_count": len(outliers),
                    "outliers": outliers[:10],  # Limit to 10 outliers in the response
                }
            
            elif method == 'zscore':
                # Z-score method
                mean = df[column].mean()
                std = df[column].std()
                z_scores = (df[column] - mean) / std
                
                # Find outliers (|z| > 3)
                outlier_indices = df[abs(z_scores) > 3].index.tolist()
                outlier_values = df.loc[outlier_indices, column].tolist()
                outlier_zscores = z_scores[outlier_indices].tolist()
                
                # Create outlier information
                for i, idx in enumerate(outlier_indices):
                    outliers.append({
                        "index": int(idx),
                        "value": float(outlier_values[i]),
                        "z_score": float(outlier_zscores[i]),
                        "type": "low" if outlier_values[i] < mean else "high",
                    })
                
                # Add insight about outliers
                if outliers:
                    self.add_analysis_insight(
                        text=f"Found {len(outliers)} outliers in column '{column}' using Z-score method.",
                        importance=3,
                        category="outliers",
                        related_columns=[column],
                    )
                
                return {
                    "success": True,
                    "method": "zscore",
                    "column": column,
                    "mean": float(mean),
                    "std": float(std),
                    "outliers_count": len(outliers),
                    "outliers": outliers[:10],  # Limit to 10 outliers in the response
                }
            
            else:
                return {
                    "success": False,
                    "message": f"Unknown outlier detection method '{method}'. Valid methods are 'iqr', 'zscore'.",
                }
        except Exception as e:
            self.shared_state.add_error(f"Failed to detect outliers: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to detect outliers: {str(e)}",
            }

    def perform_regression(self, target_column: str, feature_columns: List[str], model_type: str = 'linear') -> Dict[str, Any]:
        """Perform regression analysis.
        
        Args:
            target_column: The name of the target column.
            feature_columns: The names of the feature columns.
            model_type: The type of regression model ('linear', 'random_forest').
            
        Returns:
            The regression results.
        """
        if self.shared_state.dataframe is None:
            return {
                "success": False,
                "message": "No dataset has been loaded yet.",
            }
        
        df = self.shared_state.dataframe
        
        # Check if all columns exist
        for column in [target_column] + feature_columns:
            if column not in df.columns:
                return {
                    "success": False,
                    "message": f"Column '{column}' not found in the dataset.",
                }
        
        # Check if target column is numeric
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            return {
                "success": False,
                "message": f"Target column '{target_column}' is not numeric.",
            }
        
        try:
            # Prepare the data
            X = df[feature_columns].copy()
            y = df[target_column].copy()
            
            # Handle categorical features
            categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
            encoders = {}
            
            for column in categorical_columns:
                encoder = LabelEncoder()
                X[column] = encoder.fit_transform(X[column].astype(str))
                encoders[column] = encoder
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train the model
            if model_type == 'linear':
                model = LinearRegression()
            elif model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                return {
                    "success": False,
                    "message": f"Unknown regression model type '{model_type}'. Valid types are 'linear', 'random_forest'.",
                }
            
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Get feature importance
            if model_type == 'linear':
                feature_importance = {}
                for i, column in enumerate(feature_columns):
                    feature_importance[column] = float(abs(model.coef_[i]))
            else:
                feature_importance = {}
                for i, column in enumerate(feature_columns):
                    feature_importance[column] = float(model.feature_importances_[i])
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = [feature for feature, _ in sorted_features[:5]]
            
            # Create a predictive model object
            predictive_model = PredictiveModel(
                model_type="regression",
                target_column=target_column,
                feature_importance={k: float(v) for k, v in feature_importance.items()},
                top_features=top_features,
                metrics={
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "r2": float(r2),
                },
            )
            
            # Add the predictive model to the shared state
            self.shared_state.add_predictive_model(predictive_model)
            
            # Add insights about the regression model
            self.add_analysis_insight(
                text=f"Regression model for predicting '{target_column}' achieved R² of {r2:.2f}.",
                importance=4,
                category="predictive_modeling",
                related_columns=[target_column] + top_features,
            )
            
            if top_features:
                self.add_analysis_insight(
                    text=f"The most important feature for predicting '{target_column}' is '{top_features[0]}'.",
                    importance=4,
                    category="feature_importance",
                    related_columns=[target_column, top_features[0]],
                )
            
            return {
                "success": True,
                "model_type": model_type,
                "target_column": target_column,
                "feature_columns": feature_columns,
                "metrics": {
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "r2": float(r2),
                },
                "feature_importance": {k: float(v) for k, v in feature_importance.items()},
                "top_features": top_features,
            }
        except Exception as e:
            self.shared_state.add_error(f"Failed to perform regression: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to perform regression: {str(e)}",
            }

    def perform_classification(self, target_column: str, feature_columns: List[str], model_type: str = 'logistic') -> Dict[str, Any]:
        """Perform classification analysis.
        
        Args:
            target_column: The name of the target column.
            feature_columns: The names of the feature columns.
            model_type: The type of classification model ('logistic', 'random_forest').
            
        Returns:
            The classification results.
        """
        if self.shared_state.dataframe is None:
            return {
                "success": False,
                "message": "No dataset has been loaded yet.",
            }
        
        df = self.shared_state.dataframe
        
        # Check if all columns exist
        for column in [target_column] + feature_columns:
            if column not in df.columns:
                return {
                    "success": False,
                    "message": f"Column '{column}' not found in the dataset.",
                }
        
        try:
            # Prepare the data
            X = df[feature_columns].copy()
            y = df[target_column].copy()
            
            # Encode the target if it's categorical
            if not pd.api.types.is_numeric_dtype(y):
                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(y.astype(str))
            
            # Handle categorical features
            categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
            encoders = {}
            
            for column in categorical_columns:
                encoder = LabelEncoder()
                X[column] = encoder.fit_transform(X[column].astype(str))
                encoders[column] = encoder
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train the model
            if model_type == 'logistic':
                model = LogisticRegression(max_iter=1000, random_state=42)
            elif model_type == 'random_forest':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                return {
                    "success": False,
                    "message": f"Unknown classification model type '{model_type}'. Valid types are 'logistic', 'random_forest'.",
                }
            
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calculate precision, recall, and F1 score if binary classification
            if len(np.unique(y)) == 2:
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
            else:
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Get feature importance
            if model_type == 'logistic':
                feature_importance = {}
                for i, column in enumerate(feature_columns):
                    feature_importance[column] = float(abs(model.coef_[0][i]))
            else:
                feature_importance = {}
                for i, column in enumerate(feature_columns):
                    feature_importance[column] = float(model.feature_importances_[i])
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = [feature for feature, _ in sorted_features[:5]]
            
            # Create a predictive model object
            predictive_model = PredictiveModel(
                model_type="classification",
                target_column=target_column,
                feature_importance={k: float(v) for k, v in feature_importance.items()},
                top_features=top_features,
                metrics={
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                },
            )
            
            # Add the predictive model to the shared state
            self.shared_state.add_predictive_model(predictive_model)
            
            # Add insights about the classification model
            self.add_analysis_insight(
                text=f"Classification model for predicting '{target_column}' achieved accuracy of {accuracy:.2f}.",
                importance=4,
                category="predictive_modeling",
                related_columns=[target_column] + top_features,
            )
            
            if top_features:
                self.add_analysis_insight(
                    text=f"The most important feature for classifying '{target_column}' is '{top_features[0]}'.",
                    importance=4,
                    category="feature_importance",
                    related_columns=[target_column, top_features[0]],
                )
            
            return {
                "success": True,
                "model_type": model_type,
                "target_column": target_column,
                "feature_columns": feature_columns,
                "metrics": {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                },
                "feature_importance": {k: float(v) for k, v in feature_importance.items()},
                "top_features": top_features,
            }
        except Exception as e:
            self.shared_state.add_error(f"Failed to perform classification: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to perform classification: {str(e)}",
            }

    def add_analysis_insight(self, text: str, importance: int = 1, category: str = "", related_columns: List[str] = None) -> Dict[str, Any]:
        """Add an insight about the data analysis.
        
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

    def get_column_distribution(self, column: str) -> Dict[str, Any]:
        """Get the distribution of values in a column.
        
        Args:
            column: The name of the column.
            
        Returns:
            The distribution of values in the column.
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
            # Get the column distribution
            if pd.api.types.is_numeric_dtype(df[column]):
                # For numeric columns, calculate histogram
                hist, bin_edges = np.histogram(df[column].dropna(), bins=10)
                
                # Convert to dictionary
                distribution = {
                    "type": "numeric",
                    "histogram": {
                        "counts": hist.tolist(),
                        "bin_edges": bin_edges.tolist(),
                    },
                    "min": float(df[column].min()),
                    "max": float(df[column].max()),
                    "mean": float(df[column].mean()),
                    "median": float(df[column].median()),
                    "std": float(df[column].std()),
                    "skew": float(df[column].skew()),
                    "kurtosis": float(df[column].kurtosis()),
                }
            else:
                # For categorical columns, calculate value counts
                value_counts = df[column].value_counts().head(10)
                
                # Convert to dictionary
                distribution = {
                    "type": "categorical",
                    "value_counts": {
                        str(k): int(v) for k, v in value_counts.items()
                    },
                    "unique_values": int(df[column].nunique()),
                    "top_value": str(value_counts.index[0]) if not value_counts.empty else None,
                    "top_value_count": int(value_counts.iloc[0]) if not value_counts.empty else 0,
                }
            
            return {
                "success": True,
                "column": column,
                "distribution": distribution,
            }
        except Exception as e:
            self.shared_state.add_error(f"Failed to get column distribution: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to get column distribution: {str(e)}",
            }

    async def _process_task(self, task: Task) -> Dict[str, Any]:
        """Process a data analysis task.
        
        Args:
            task: The task to process.
            
        Returns:
            The result of processing the task.
        """
        if self.shared_state.dataframe is None:
            return {
                "success": False,
                "message": "No dataset has been loaded yet. Cannot analyze data.",
            }
        
        # Create a prompt for the model to analyze the data
        df = self.shared_state.dataframe
        
        # Get a sample of the data
        sample = df.head(5).to_string()
        
        # Get the dataset info
        dataset_info = self.shared_state.dataset_info.model_dump()
        
        prompt = f"""
        You are a data analysis agent. Your task is to analyze the dataset and generate insights.
        
        Dataset information:
        - Rows: {dataset_info['rows']}
        - Columns: {dataset_info['columns']}
        - Column names: {', '.join(dataset_info['column_names'])}
        - Missing values: {dataset_info['missing_values']}
        
        Here's a sample of the data:
        {sample}
        
        Please analyze the dataset and generate insights. For each insight, explain why it's important and what it means for the data.
        
        You have the following tools available:
        - calculate_correlation: Calculate correlation between numeric columns
        - calculate_summary_statistics: Calculate summary statistics for numeric columns
        - detect_outliers: Detect outliers in a numeric column
        - perform_regression: Perform regression analysis
        - perform_classification: Perform classification analysis
        - add_analysis_insight: Add an insight about the data analysis
        - get_column_distribution: Get the distribution of values in a column
        
        For each analysis operation, call the appropriate tool with the necessary parameters.
        """
        
        # Run the model to generate analysis operations
        response = await self.run(prompt)
        
        # Return the results
        return {
            "success": True,
            "message": "Data analysis completed successfully.",
            "model_response": response.get("text", ""),
            "function_calls": response.get("function_call", {}),
            "function_responses": response.get("function_response", {}),
            "insights": [insight.model_dump() for insight in self.shared_state.insights],
            "predictive_models": [model.model_dump() for model in self.shared_state.predictive_models],
        }