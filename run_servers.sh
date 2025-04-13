#!/bin/bash

# Start the backend server
echo "Starting backend server on port 12001..."
cd /workspace/DataHammerNoLlamaIndex
python3 backend/main.py &
BACKEND_PID=$!

# Wait for the backend to start
sleep 2

# Start the frontend server
echo "Starting frontend server on port 12000..."
cd /workspace/DataHammerNoLlamaIndex
npm run dev &
FRONTEND_PID=$!

# Function to handle script termination
cleanup() {
    echo "Stopping servers..."
    kill $BACKEND_PID
    kill $FRONTEND_PID
    exit 0
}

# Register the cleanup function for when the script is terminated
trap cleanup SIGINT SIGTERM

# Keep the script running
echo "Both servers are running. Press Ctrl+C to stop."
wait