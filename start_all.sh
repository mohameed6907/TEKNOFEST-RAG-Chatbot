#!/bin/bash

# start_all.sh
# Bash script to run both LangGraph Dev server and FastAPI Web App inside Linux/Docker container

echo "Starting LangGraph Dev Server (LangGraph Studio)..."
langgraph dev --host 0.0.0.0 --port 2024 &
LANGGRAPH_PID=$!

echo "Waiting 5 seconds for LangGraph server to initialize..."
sleep 5

echo "Starting FastAPI Chat Server..."
uvicorn app.main:app --host 0.0.0.0 --port 8010 --reload &
FASTAPI_PID=$!

# Handle exit and kill both processes
cleanup() {
    echo "Stopping all services..."
    kill $LANGGRAPH_PID 2>/dev/null
    kill $FASTAPI_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "Both servers are running!"
echo "- LangGraph Dev Server on port 2024"
echo "- FastAPI Server on port 8010"
echo "Press Ctrl+C to stop both."

# Keep script running
wait
