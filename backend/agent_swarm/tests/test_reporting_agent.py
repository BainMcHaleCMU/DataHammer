"""
Test module for the ReportingTaskAgent and ReportingAgent.

This module contains tests for the reporting agent functionality.
"""

import os
import unittest
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock

from ..llama_workflow.reporting_agent import ReportingTaskAgent
from ..agents.reporting_agent import ReportingAgent


class TestReportingTaskAgent(unittest.TestCase):
    """Test cases for the ReportingTaskAgent."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        self.agent = ReportingTaskAgent(output_dir=self.test_dir)

    def tearDown(self):
        """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    def test_init(self):
        """Test initialization of the agent."""
        self.assertEqual(self.agent.name, "ReportingAgent")
        self.assertEqual(self.agent.output_dir, self.test_dir)
        self.assertTrue(os.path.exists(self.test_dir))

    def test_validate_input_empty(self):
        """Test input validation with empty input."""
        input_data = {}
        validated = self.agent._validate_input(input_data)
        
        self.assertIn("environment", validated)
        self.assertIn("goals", validated)
        self.assertIn("report_format", validated)
        self.assertIn("sections", validated)
        
        self.assertEqual(validated["report_format"], "jupyter")
        self.assertEqual(validated["goals"], ["Generate comprehensive data analysis report"])
        self.assertEqual(validated["sections"], [])

    def test_validate_input_invalid_format(self):
        """Test input validation with invalid report format."""
        input_data = {"report_format": "invalid_format"}
        validated = self.agent._validate_input(input_data)
        
        self.assertEqual(validated["report_format"], "jupyter")

    def test_safe_get(self):
        """Test safe dictionary access."""
        test_dict = {
            "level1": {
                "level2": {
                    "level3": "value"
                }
            }
        }
        
        # Test valid path
        result = self.agent._safe_get(test_dict, ["level1", "level2", "level3"])
        self.assertEqual(result, "value")
        
        # Test invalid path
        result = self.agent._safe_get(test_dict, ["level1", "invalid", "level3"])
        self.assertIsNone(result)
        
        # Test with default value
        result = self.agent._safe_get(test_dict, ["invalid"], "default")
        self.assertEqual(result, "default")

    def test_generate_report_path(self):
        """Test report path generation."""
        # Test jupyter format
        path = self.agent._generate_report_path("jupyter")
        self.assertTrue(path.endswith(".ipynb"))
        self.assertTrue(path.startswith(self.test_dir))
        
        # Test other format
        path = self.agent._generate_report_path("markdown")
        self.assertTrue(path.endswith(".markdown"))

    def test_handle_insights_empty(self):
        """Test insights handling with empty data."""
        cells = self.agent._handle_insights({})
        
        self.assertEqual(len(cells), 2)
        self.assertEqual(cells[1]["source"], "*No insights available*")

    def test_handle_insights_valid(self):
        """Test insights handling with valid data."""
        insights = {
            "dataset1": [
                {
                    "type": "Correlation",
                    "description": "Strong correlation between X and Y",
                    "details": "r=0.95"
                }
            ]
        }
        
        cells = self.agent._handle_insights(insights)
        
        self.assertEqual(len(cells), 2)
        self.assertIn("dataset1", cells[1]["source"])
        self.assertIn("Strong correlation between X and Y", cells[1]["source"])

    @patch('agent_swarm.custom_framework.settings.Settings')
    def test_run_minimal_input(self, mock_settings):
        """Test run method with minimal input."""
        # Mock the LLM to avoid actual API calls
        mock_llm = MagicMock()
        mock_settings.llm = mock_llm
        
        # Run with minimal input
        result = self.agent.run({"environment": {}})
        
        # Check that we got a valid result structure
        self.assertIn("JupyterLogbook", result)
        self.assertIn("Report.summary", result)
        self.assertIn("Report.recommendations", result)
        self.assertIn("Report.path", result)
        
        # Check that the report path exists
        self.assertTrue(os.path.exists(self.test_dir))

    @patch('agent_swarm.custom_framework.settings.Settings')
    def test_run_error_handling(self, mock_settings):
        """Test error handling in run method."""
        # Mock the LLM to raise an exception
        mock_llm = MagicMock()
        mock_settings.llm = mock_llm
        
        # Create a method that will raise an exception
        def raise_exception(*args, **kwargs):
            raise ValueError("Test exception")
        
        # Patch the _safe_get method to raise an exception
        with patch.object(self.agent, '_safe_get', side_effect=raise_exception):
            result = self.agent.run({"environment": {}})
            
            # Check that we got error information
            self.assertIn("error", result)
            self.assertIn("stack_trace", result)
            self.assertEqual(result["Report.summary"]["title"], "Error Report")


class TestReportingAgent(unittest.TestCase):
    """Test cases for the ReportingAgent."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        self.agent = ReportingAgent(output_dir=self.test_dir)

    def tearDown(self):
        """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    def test_init(self):
        """Test initialization of the agent."""
        self.assertEqual(self.agent.name, "ReportingAgent")
        self.assertEqual(self.agent.output_dir, self.test_dir)
        self.assertIsInstance(self.agent.task_agent, ReportingTaskAgent)

    @patch.object(ReportingTaskAgent, 'run')
    def test_run_success(self, mock_run):
        """Test successful run."""
        # Mock the task agent's run method
        mock_run.return_value = {
            "Report.path": "/path/to/report.ipynb",
            "Report.content": {"title": "Test Report"},
            "Report.summary": {"datasets_analyzed": ["dataset1", "dataset2"]}
        }
        
        # Run the agent
        result = self.agent.run({})
        
        # Check the result
        self.assertEqual(result["report"], "/path/to/report.ipynb")
        self.assertEqual(result["report_content"]["title"], "Test Report")
        
        # Check that visualization requests were generated
        self.assertEqual(len(result["visualization_requests"]), 3)
        self.assertEqual(result["visualization_requests"][0]["type"], "summary_plot")
        self.assertEqual(result["visualization_requests"][1]["data"], "dataset1")
        self.assertEqual(result["visualization_requests"][2]["data"], "dataset2")

    @patch.object(ReportingTaskAgent, 'run')
    def test_run_task_error(self, mock_run):
        """Test handling of task agent errors."""
        # Mock the task agent's run method to return an error
        mock_run.return_value = {
            "error": "Task error",
            "Report.content": {"error": "Task error"}
        }
        
        # Run the agent
        result = self.agent.run({})
        
        # Check the result
        self.assertEqual(result["error"], "Task error")
        self.assertIsNone(result["report"])
        self.assertEqual(len(result["visualization_requests"]), 0)

    @patch.object(ReportingTaskAgent, 'run')
    def test_run_exception(self, mock_run):
        """Test handling of exceptions."""
        # Mock the task agent's run method to raise an exception
        mock_run.side_effect = ValueError("Test exception")
        
        # Run the agent
        result = self.agent.run({})
        
        # Check the result
        self.assertIn("Error generating report", result["error"])
        self.assertIn("stack_trace", result)
        self.assertIsNone(result["report"])
        self.assertEqual(len(result["visualization_requests"]), 0)

    def test_parameter_validation(self):
        """Test validation of parameters."""
        # Create a mock task agent to avoid actual execution
        with patch.object(self.agent, 'task_agent'):
            # Test invalid report format
            with patch.object(self.agent, 'logger') as mock_logger:
                self.agent.run({}, report_format="invalid")
                mock_logger.warning.assert_called_with(
                    "Unsupported report format: invalid, defaulting to jupyter"
                )
            
            # Test invalid sections
            with patch.object(self.agent, 'logger') as mock_logger:
                self.agent.run({}, sections="not_a_list")
                mock_logger.warning.assert_called_with(
                    "Sections parameter is not a list, using empty list"
                )
            
            # Test invalid goals
            with patch.object(self.agent, 'logger') as mock_logger:
                self.agent.run({}, goals=[])
                mock_logger.warning.assert_called_with(
                    "Goals parameter is invalid, using default goal"
                )


if __name__ == '__main__':
    unittest.main()