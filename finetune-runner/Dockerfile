FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential cmake curl \
    ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

# PyTorch with CUDA 12.4 (2.6+ required for torchao compatibility)
RUN pip3 install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu124

# Unsloth (brings compatible transformers/trl/peft)
RUN pip3 install --no-cache-dir unsloth

# Remaining training deps
RUN pip3 install --no-cache-dir \
    datasets bitsandbytes accelerate sentencepiece protobuf

# Verify basic Python + pip install succeeded (GPU packages verified at runtime)
RUN python3 -c "import json, hashlib, signal, sys; print('base imports ok')" && \
    pip3 list | grep -E "unsloth|torch|trl|peft|datasets|bitsandbytes" && \
    echo "ALL PACKAGES INSTALLED"

COPY train.py /app/train.py
WORKDIR /work
ENTRYPOINT ["python3", "-u", "/app/train.py"]
