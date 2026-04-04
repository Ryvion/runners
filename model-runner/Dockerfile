# Ryvion Custom Model Runner
# Supports: GGUF (llama.cpp), ONNX (onnxruntime), PyTorch/SafeTensors (transformers)
#
# Build:
#   docker build -t ghcr.io/ryvion/model-runner:latest .
#
# GPU build:
#   docker build -f Dockerfile.gpu -t ghcr.io/ryvion/model-runner:gpu .

FROM python:3.11-slim

WORKDIR /app

# System deps for model loading and building llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU from dedicated index
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# All other deps from PyPI
RUN pip install --no-cache-dir \
    numpy \
    onnxruntime \
    llama-cpp-python \
    transformers \
    safetensors \
    sentencepiece \
    protobuf

COPY run.py /app/run.py

# Security: run as non-root
RUN useradd -m -s /bin/bash runner
USER runner

ENTRYPOINT ["python", "/app/run.py"]
