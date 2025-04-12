#!/bin/bash

# Start the FastAPI backend
cd /workspace/data-analytics-app/backend
python main.py &
BACKEND_PID=$!

# Start the Next.js frontend
cd /workspace/data-analytics-app
npm run dev -- -p 12000 --hostname 0.0.0.0 &
FRONTEND_PID=$!

# Handle cleanup on exit
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT TERM EXIT

# Keep the script running
wait