#!/bin/bash
set -e

echo "============================================"
echo "  TEKNOFEST RAG Chatbot - Starting up..."
echo "============================================"

# Step 1: Ingest local documents (if any exist in RAG/raw/)
echo ""
echo "[1/3] Ingesting local documents..."
if [ "$(find /app/RAG/raw -type f \( -name '*.pdf' -o -name '*.docx' -o -name '*.txt' -o -name '*.md' \) 2>/dev/null | head -1)" ]; then
    python -m scripts.ingest_local_docs
    echo "  -> Local document indexing completed."
else
    echo "  -> No local documents found in RAG/raw/, skipping."
fi

# Step 2: Crawl TEKNOFEST site (only if index doesn't exist yet)
echo ""
echo "[2/3] Checking TEKNOFEST site index..."
if [ ! -d "/app/RAG/chroma_teknofest_site/chroma.sqlite3" ] && [ -z "$(ls -A /app/RAG/chroma_teknofest_site/ 2>/dev/null)" ]; then
    echo "  -> No existing site index found. Crawling teknofest.org..."
    python -m scripts.crawl_teknofest || echo "  -> WARNING: Crawl failed (network issue?), continuing without site index."
    echo "  -> Site crawl completed."
else
    echo "  -> Existing site index found, skipping crawl."
fi

# Step 3: Start the FastAPI application
echo ""
echo "[3/3] Starting FastAPI server..."
echo "============================================"
echo "  App available at: http://0.0.0.0:8000"
echo "============================================"

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
