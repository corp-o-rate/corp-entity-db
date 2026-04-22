# RunPod Serverless Dockerfile for Entity Database Search
# Lighter image - no T5-Gemma or GLiNER2 needed, CPU-only
FROM python:3.11-slim

WORKDIR /app

# Install system deps (build-essential for native extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only first for smaller image
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# HuggingFace token for gated model access (needed for both DB download and model download)
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Pre-download database + USearch indexes (shard manifests, shards, monolithic fallback) from HuggingFace
COPY predownload.py .
RUN python predownload.py

# Pre-download embedding model (google/embeddinggemma-300m)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('google/embeddinggemma-300m'); print('Embedding model downloaded')"

# Copy handler
COPY handler.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the handler
CMD ["python", "-u", "handler.py"]
