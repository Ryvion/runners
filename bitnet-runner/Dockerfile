# BitNet 1-bit LLM inference runner for Ryvion.
# CPU-only inference using Microsoft's bitnet.cpp framework.
# Produces build/bin/llama-cli + quantized GGUF at build time.

FROM python:3.11-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential clang ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Clone bitnet.cpp (with submodules for llama.cpp)
RUN git clone --recursive --depth 1 https://github.com/microsoft/BitNet.git /opt/bitnet
WORKDIR /opt/bitnet

# Install Python deps (needed for setup_env.py codegen + gguf conversion)
RUN pip install --no-cache-dir -r requirements.txt

# Download the pre-built GGUF model
RUN huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf \
    --local-dir models/BitNet-b1.58-2B-4T

# Run setup_env.py: generates optimised kernels, compiles llama-cli,
# and quantises the model to i2_s format.
RUN python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy compiled binary and quantised model only
COPY --from=builder /opt/bitnet/build/bin /opt/bitnet/build/bin
COPY --from=builder /opt/bitnet/models /opt/bitnet/models

RUN useradd -r -s /usr/sbin/nologin runner
COPY run.py /app/run.py

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import sys; sys.exit(0)" || exit 1

STOPSIGNAL SIGTERM

USER runner
WORKDIR /work
ENTRYPOINT ["python", "-u", "/app/run.py"]
