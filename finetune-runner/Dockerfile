FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential cmake curl \
    ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support first
RUN pip3 install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu124

# Install training stack (without compiling custom CUDA kernels)
RUN pip3 install --no-cache-dir \
    unsloth datasets trl transformers peft \
    bitsandbytes accelerate sentencepiece protobuf

# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip libgomp1 ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.10/dist-packages \
     /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY train.py /app/train.py
WORKDIR /work
ENTRYPOINT ["python3", "-u", "/app/train.py"]
