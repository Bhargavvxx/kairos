# --- Stage 1: Dependency Installation Stage ---
# Use a standard Python image to install our dependencies.
FROM python:3.10-slim as builder

# Set the working directory inside the container.
WORKDIR /app

# Install system dependencies that might be needed for building Python packages.
RUN apt-get update && apt-get install -y \
    gcc g++ \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to latest version
RUN pip install --upgrade pip wheel setuptools

# Copy the file that lists our project's dependencies.
COPY ./backend/requirements.txt .

# Install PyTorch first (CPU version for smaller size)
RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu

# Install the remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the ML model to the builder stage.
# This saves 5-10 seconds of download time on every container startup.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/tmp/models')"

# --- Stage 2: Final Application Stage ---
# Use the same slim Python image for our final, clean application.
FROM python:3.10-slim

# Set the working directory.
WORKDIR /app

# Install only the necessary runtime libraries to keep the image small.
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash appuser

# Copy installed Python dependencies from the builder stage.
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the pre-downloaded model cache from the builder stage
COPY --from=builder /tmp/models /home/appuser/.cache/torch/sentence_transformers
RUN chown -R appuser:appuser /home/appuser/.cache

# Create necessary directories for the app
RUN mkdir -p /app/chroma_db /app/cache && \
    chown -R appuser:appuser /app

# Switch to the non-root user for better security.
USER appuser

# Copy our application's source code into the container.
COPY --chown=appuser:appuser ./backend/app ./app

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=2

# Expose the port that our FastAPI application will run on.
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# The command to start the Uvicorn server when the container launches.
# Optimized for HackRx performance
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--loop", "uvloop", "--access-log"]