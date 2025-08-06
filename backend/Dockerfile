# --- Stage 1: Dependency Installation Stage ---
# Use a standard Python image to install our dependencies.
FROM python:3.10-slim as builder

# Set the working directory inside the container.
WORKDIR /app

# Install system dependencies that might be needed for building Python packages.
RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy the file that lists our project's dependencies.
COPY ./backend/requirements.txt .

# Install the dependencies listed in requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the ML model to the builder stage.
# This saves 5-10 seconds of download time on every container startup.
#RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# --- Stage 2: Final Application Stage ---
# Use the same slim Python image for our final, clean application.
FROM python:3.10-slim

# Set the working directory.
WORKDIR /app

# Install only the necessary runtime libraries to keep the image small.
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home appuser

# Copy installed Python dependencies from the builder stage.
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the pre-downloaded model cache from the builder stage to the appuser's home directory.
#COPY --from=builder /root/.cache /home/appuser/.cache
# Ensure the new user owns the cached files.
# RUN chown -R appuser:appuser /home/appuser/.cache    <--- COMMENT THIS LINE

# Switch to the non-root user for better security.
USER appuser

# Copy our application's source code into the container.
COPY --chown=appuser:appuser ./backend/app ./app

# Expose the port that our FastAPI application will run on.
EXPOSE 8000

# The command to start the Uvicorn server when the container launches.
# We use uvloop for a significant performance boost over the default event loop.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2