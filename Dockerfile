FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    gcc \
    g++ \
    poppler-utils \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from backend directory
COPY backend/requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ML models to avoid Railway timeout
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "import nltk; nltk.download('punkt', quiet=True)"

# Copy the entire backend directory
COPY backend/ .

# Create data directories
RUN mkdir -p chroma_db graph_db cache logs

# Railway uses PORT environment variable
EXPOSE $PORT

# Start the FastAPI app
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT