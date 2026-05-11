# -----------------------------------------------------------------
# TEKNOFEST RAG Chatbot – Docker image
# Base: 36847439/agent_25112025:latest
# -----------------------------------------------------------------
FROM 36847439/agent_25112025:latest

# Set working directory inside the container
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Ensure RAG directories exist
RUN mkdir -p /app/RAG/raw /app/RAG/chroma_local_docs /app/RAG/chroma_teknofest_site

# Ensure scripts package is importable
RUN touch /app/scripts/__init__.py 2>/dev/null || true

# Copy and set entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose the FastAPI port
EXPOSE 8000

# Entrypoint runs indexing + starts uvicorn
ENTRYPOINT ["/app/entrypoint.sh"]
