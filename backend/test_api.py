"""
Test script for the DataHammer API.
"""

import requests
import time
import sys
import os
from pathlib import Path

# URL of the API
API_URL = "http://localhost:8000"

def test_root():
    """Test the root endpoint."""
    response = requests.get(f"{API_URL}/")
    print("Root endpoint response:", response.json())
    assert response.status_code == 200
    assert "message" in response.json()

def test_analyze(file_path):
    """Test the analyze endpoint."""
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f)}
        response = requests.post(f"{API_URL}/analyze", files=files)
    
    print("Analyze endpoint response status:", response.status_code)
    if response.status_code == 200:
        print("Analyze endpoint response:", response.json())
    else:
        print("Analyze endpoint error:", response.text)
    
    assert response.status_code == 200
    return response.json()

def test_status():
    """Test the status endpoint."""
    response = requests.get(f"{API_URL}/status")
    print("Status endpoint response:", response.json())
    assert response.status_code == 200
    return response.json()

def test_results():
    """Test the results endpoint."""
    response = requests.get(f"{API_URL}/results")
    print("Results endpoint response:", response.json())
    assert response.status_code == 200
    return response.json()

def main():
    """Run the tests."""
    # Check if a file path was provided
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    # Test the root endpoint
    test_root()
    
    # Test the analyze endpoint
    test_analyze(file_path)
    
    # Wait for the pipeline to start
    print("Waiting for the pipeline to start...")
    time.sleep(5)
    
    # Test the status endpoint
    status = test_status()
    
    # Wait for the pipeline to complete
    while status["pipeline_status"]["state"] != "idle" and status["pipeline_status"]["pending_tasks"] > 0:
        print("Pipeline is still running. Waiting...")
        time.sleep(5)
        status = test_status()
    
    # Test the results endpoint
    results = test_results()
    
    print("All tests passed!")

if __name__ == "__main__":
    main()