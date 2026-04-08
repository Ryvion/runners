# BitNet 1-bit LLM inference runner for Ryvion.
# CPU-only inference using Microsoft's bitnet.cpp framework.

FROM python:3.11-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential clang ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Clone bitnet.cpp with llama.cpp submodule
RUN git clone --recursive --depth 1 https://github.com/microsoft/BitNet.git /opt/bitnet
WORKDIR /opt/bitnet

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# setup_env.py downloads the model from HuggingFace, converts to GGUF,
# generates optimised kernels, and compiles llama-cli.
# Use the official HF model ID (NOT the pre-converted gguf repo).
RUN python setup_env.py -md microsoft/BitNet-b1.58-2B-4T -q i2_s

# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy compiled binary and model
COPY --from=builder /opt/bitnet/build/bin /opt/bitnet/build/bin
COPY --from=builder /opt/bitnet/models /opt/bitnet/models

RUN useradd -r -s /usr/sbin/nologin runner
COPY run.py /app/run.py

STOPSIGNAL SIGTERM

USER runner
WORKDIR /work
ENTRYPOINT ["python", "-u", "/app/run.py"]
