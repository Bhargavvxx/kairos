FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y curl build-essential gcc poppler-utils && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
COPY . .
RUN mkdir -p chroma_db graph_db cache logs
EXPOSE $PORT
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT