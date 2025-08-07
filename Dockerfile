FROM python:3.11-slim

WORKDIR /app

# Install only essential packages
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY backend/ .

# Create directories
RUN mkdir -p chroma_db graph_db cache logs

EXPOSE $PORT
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT