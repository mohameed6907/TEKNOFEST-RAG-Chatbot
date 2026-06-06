#!/bin/bash

# Clean up problematic empty env vars inherited from container
unset OPENAI_BASE_URL

# FastAPI RAG Chatbot
cd /data && uvicorn app.main:app --host 0.0.0.0 --port 8010 --reload &
