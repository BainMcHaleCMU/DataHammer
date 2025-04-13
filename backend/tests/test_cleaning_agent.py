"""
Tests for the CleaningAgent and CleaningTaskAgent classes.
"""

import unittest
from unittest.mock import MagicMock, patch
import logging

from backend.agent_swarm.agents.cleaning_agent import CleaningAgent
from backend.agent_swarm.llama_workflow.cleaning_agent import CleaningTaskAgent, CleaningStrategy


class TestCleaningTaskAgent(unittest.TestCase):
    """Test cases for the CleaningTaskAgent class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the LLM
        self.mock_llm = MagicMock()
        
        # Create a test instance with the mock LLM
        with patch('backend.agent_swarm.llama_workflow.base.Settings') as mock_settings:
            mock_settings.llm = self.mock_llm
            self.agent = CleaningTaskAgent(llm=self.mock_llm)
        
        # Sample test data
        self.test_environment = {
            "Data Overview": {
                "raw_data": {
                    "test_dataset": {
                        "type": "dataframe",
                        "rows": 100,
                        "columns": ["col1", "col2", "col3"],
                        "source": "test_source"
                    },
                    "error_dataset": {
                        "type": "error",
                        "error": "Failed to load dataset"
                    }
                },
                "summary": {
                    "test_dataset": {
                        "has_missing_values": True,
                        "has_duplicates": True,
                        "duplicate_count": 5
                    }
                },
                "statistics": {
                    "test_dataset": {
                        "numeric_columns": {
                            "col1": {"mean": 10, "std": 2},
                            "col2": {"mean": 20, "std": 5}
                        },
                        "categorical_columns": {
                            "col3": {"unique_values": 3}
                        }
                    }
                }
            }
        }
        
        self.test_input = {
            "environment": self.test_environment,
            "goals": ["Clean test data"]
        }

    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.name, "CleaningAgent")
        self.assertEqual(self.agent.llm, self.mock_llm)
        
        # Check default strategies
        self.assertTrue(self.agent.default_strategies[CleaningStrategy.MISSING_VALUES.value])
        self.assertTrue(self.agent.default_strategies[CleaningStrategy.DUPLICATES.value])
        self.assertTrue(self.agent.default_strategies[CleaningStrategy.OUTLIERS.value])
        self.assertTrue(self.agent.default_strategies[CleaningStrategy.DATA_TYPES.value])
        self.assertFalse(self.agent.default_strategies[CleaningStrategy.NORMALIZATION.value])
        self.assertFalse(self.agent.default_strategies[CleaningStrategy.ENCODING.value])

    def test_validate_input_data(self):
        """Test input data validation."""
        # Valid input
        errors = self.agent._validate_input_data(self.test_input)
        self.assertEqual(len(errors), 0)
        
        # Missing environment
        errors = self.agent._validate_input_data({})
        self.assertEqual(len(errors), 1)
        self.assertIn("Missing 'environment'", errors[0])
        
        # Missing raw data
        invalid_input = {
            "environment": {
                "Data Overview": {
                    "summary": {},
                    "statistics": {}
                }
            }
        }
        errors = self.agent._validate_input_data(invalid_input)
        self.assertEqual(len(errors), 1)
        self.assertIn("No raw data found", errors[0])

    def test_get_cleaning_strategies(self):
        """Test getting cleaning strategies."""
        # Default strategies
        strategies = self.agent._get_cleaning_strategies(self.test_input)
        self.assertEqual(len(strategies), 6)
        self.assertTrue(strategies[CleaningStrategy.MISSING_VALUES.value])
        
        # Custom strategies
        custom_input = self.test_input.copy()
        custom_input["cleaning_strategies"] = {
            CleaningStrategy.MISSING_VALUES.value: False,
            CleaningStrategy.NORMALIZATION.value: True,
            "invalid_strategy": True  # Should be ignored
        }
        
        strategies = self.agent._get_cleaning_strategies(custom_input)
        self.assertEqual(len(strategies), 6)
        self.assertFalse(strategies[CleaningStrategy.MISSING_VALUES.value])
        self.assertTrue(strategies[CleaningStrategy.NORMALIZATION.value])
        self.assertNotIn("invalid_strategy", strategies)

    def test_apply_cleaning_strategy(self):
        """Test applying cleaning strategies."""
        dataset_name = "test_dataset"
        dataset_info = self.test_environment["Data Overview"]["raw_data"][dataset_name]
        dataset_summary = self.test_environment["Data Overview"]["summary"][dataset_name]
        dataset_stats = self.test_environment["Data Overview"]["statistics"][dataset_name]
        
        # Test missing values strategy
        result = self.agent._apply_cleaning_strategy(
            CleaningStrategy.MISSING_VALUES.value,
            dataset_name,
            dataset_info,
            dataset_summary,
            dataset_stats
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result["operation"], CleaningStrategy.MISSING_VALUES.value)
        self.assertEqual(result["status"], "success")
        self.assertIn("col1", result["affected_columns"])
        
        # Test strategy that doesn't apply
        dataset_summary_no_duplicates = dataset_summary.copy()
        dataset_summary_no_duplicates["has_duplicates"] = False
        
        result = self.agent._apply_cleaning_strategy(
            CleaningStrategy.DUPLICATES.value,
            dataset_name,
            dataset_info,
            dataset_summary_no_duplicates,
            dataset_stats
        )
        
        self.assertIsNone(result)
        
        # Test custom strategy
        custom_strategy = {
            "description": "Custom cleaning operation",
            "affected_columns": ["col1"],
            "details": "Applied custom logic"
        }
        
        result = self.agent._apply_cleaning_strategy(
            CleaningStrategy.CUSTOM.value,
            dataset_name,
            dataset_info,
            dataset_summary,
            dataset_stats,
            custom_strategy
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result["operation"], CleaningStrategy.CUSTOM.value)
        self.assertEqual(result["description"], "Custom cleaning operation")

    def test_run_with_valid_data(self):
        """Test running the agent with valid data."""
        result = self.agent.run(self.test_input)
        
        # Check result structure
        self.assertIn("Cleaned Data.processed_data", result)
        self.assertIn("Cleaned Data.cleaning_steps", result)
        self.assertIn("Cleaned Data.validation", result)
        
        # Check processed data
        processed_data = result["Cleaned Data.processed_data"]
        self.assertIn("test_dataset", processed_data)
        self.assertEqual(processed_data["test_dataset"]["type"], "dataframe")
        self.assertEqual(processed_data["test_dataset"]["is_cleaned"], True)
        
        # Check cleaning steps
        cleaning_steps = result["Cleaned Data.cleaning_steps"]
        self.assertIn("test_dataset", cleaning_steps)
        self.assertGreater(len(cleaning_steps["test_dataset"]), 0)
        
        # Check validation results
        validation = result["Cleaned Data.validation"]
        self.assertIn("test_dataset", validation)
        self.assertEqual(validation["test_dataset"]["status"], "success")
        
        # Check error dataset handling
        self.assertIn("error_dataset", validation)
        self.assertEqual(validation["error_dataset"]["status"], "skipped")

    def test_run_with_invalid_data(self):
        """Test running the agent with invalid data."""
        # Empty environment
        invalid_input = {"environment": {}}
        result = self.agent.run(invalid_input)
        
        self.assertIn("Cleaned Data.validation", result)
        validation = result["Cleaned Data.validation"]
        self.assertEqual(validation["status"], "error")
        self.assertGreater(len(validation["errors"]), 0)


class TestCleaningAgent(unittest.TestCase):
    """Test cases for the CleaningAgent class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock for the task agent
        self.mock_task_agent = MagicMock()
        
        # Create the agent with the mock task agent
        self.agent = CleaningAgent()
        self.agent.task_agent = self.mock_task_agent
        
        # Sample environment
        self.test_environment = {"loaded_data": "test_data_reference"}

    def test_validate_data_reference(self):
        """Test data reference validation."""
        self.assertTrue(self.agent._validate_data_reference("valid_reference"))
        self.assertFalse(self.agent._validate_data_reference(None))
        self.assertFalse(self.agent._validate_data_reference(""))

    def test_prepare_cleaning_strategies(self):
        """Test preparing cleaning strategies."""
        # Default strategies
        strategies = self.agent._prepare_cleaning_strategies({})
        self.assertEqual(len(strategies), 4)
        self.assertTrue(strategies[CleaningStrategy.MISSING_VALUES.value])
        
        # User strategies
        user_strategies = {
            CleaningStrategy.MISSING_VALUES.value: False,
            CleaningStrategy.CUSTOM.value: True
        }
        
        strategies = self.agent._prepare_cleaning_strategies(user_strategies)
        self.assertEqual(len(strategies), 5)
        self.assertFalse(strategies[CleaningStrategy.MISSING_VALUES.value])
        self.assertTrue(strategies[CleaningStrategy.CUSTOM.value])

    def test_run_with_missing_data_reference(self):
        """Test running with missing data reference."""
        result = self.agent.run({})
        
        self.assertEqual(result["status"], "error")
        self.assertIn("Invalid or missing data reference", result["error"])
        self.assertIsNone(result["cleaned_data"])

    def test_run_with_valid_data(self):
        """Test running with valid data."""
        # Mock the task agent response
        mock_result = {
            "Cleaned Data.processed_data": {"test_dataset": {"type": "dataframe"}},
            "Cleaned Data.cleaning_steps": {"test_dataset": [{"operation": "test"}]},
            "Cleaned Data.validation": {"test_dataset": {"status": "success"}}
        }
        self.mock_task_agent.run.return_value = mock_result
        
        # Run the agent
        result = self.agent.run(self.test_environment)
        
        # Verify task agent was called correctly
        self.mock_task_agent.run.assert_called_once()
        task_input = self.mock_task_agent.run.call_args[0][0]
        self.assertEqual(task_input["environment"], self.test_environment)
        self.assertEqual(task_input["data_reference"], "test_data_reference")
        
        # Check result
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["cleaned_data"], mock_result["Cleaned Data.processed_data"])
        self.assertEqual(result["cleaning_steps"], mock_result["Cleaned Data.cleaning_steps"])
        self.assertEqual(result["validation"], mock_result["Cleaned Data.validation"])
        self.assertGreater(len(result["suggestions"]), 0)

    def test_run_with_task_agent_error(self):
        """Test handling task agent errors."""
        # Mock the task agent to raise an exception
        self.mock_task_agent.run.side_effect = ValueError("Test error")
        
        # Run the agent
        result = self.agent.run(self.test_environment)
        
        # Check error handling
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "Test error")
        self.assertIsNone(result["cleaned_data"])
        self.assertEqual(result["cleaning_steps"], {})
        self.assertEqual(result["validation"]["status"], "error")
        self.assertGreater(len(result["suggestions"]), 0)


if __name__ == "__main__":
    unittest.main()