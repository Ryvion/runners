# BitNet 1-bit LLM inference runner for Ryvion.
# CPU-only inference using Microsoft's bitnet.cpp framework.

FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential clang-18 ca-certificates \
    python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/clang-18 /usr/bin/clang && \
    ln -sf /usr/bin/clang++-18 /usr/bin/clang++

# Clone bitnet.cpp with llama.cpp submodule
ARG BITNET_REF=01eb415772c342d9f20dc42772f1583ae1e5b102
ARG BITNET_MODEL_REV=a1f2f1c765812aa8af3f6eda4a313707064bba15
ARG BITNET_MODEL_SHA256=4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162
RUN git clone --depth 1 https://github.com/microsoft/BitNet.git /opt/bitnet && \
    cd /opt/bitnet && \
    git fetch --depth 1 origin "${BITNET_REF}" && \
    git checkout --detach "${BITNET_REF}" && \
    git submodule update --init --recursive --depth 1
WORKDIR /opt/bitnet

# Install Python deps in venv
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt
ENV PATH="/opt/venv/bin:${PATH}"

# Patch upstream const-correctness error in the AVX2 kernel before setup_env.py
# compiles llama-cli. Keep this explicit so upstream source drift fails loudly.
RUN python3 -c 'from pathlib import Path; p = Path("src/ggml-bitnet-mad.cpp"); s = p.read_text(); old = "        int8_t * y_col = y + col * by;"; new = "        const int8_t * y_col = y + col * by;"; assert old in s, "expected BitNet source pattern not found"; p.write_text(s.replace(old, new))' && \
    grep -F "const int8_t * y_col = y + col * by;" src/ggml-bitnet-mad.cpp

# Download the pre-converted GGUF; BitNet's converter does not support the
# raw BitNetForCausalLM HF architecture yet.
RUN huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf \
    --revision "${BITNET_MODEL_REV}" \
    --include "ggml-model-i2_s.gguf" \
    --local-dir models/BitNet-b1.58-2B-4T && \
    echo "${BITNET_MODEL_SHA256}  models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf" | sha256sum -c -

# setup_env.py: generates kernels and compiles llama-cli, then reuses the GGUF.
RUN /opt/venv/bin/python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s || \
    (cat logs/*.log 2>/dev/null; exit 1)

# ---------------------------------------------------------------------------
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 libgomp1 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy compiled binary and model
COPY --from=builder /opt/bitnet/build/bin /opt/bitnet/build/bin
COPY --from=builder /opt/bitnet/models /opt/bitnet/models

RUN useradd -r -s /usr/sbin/nologin runner
COPY run.py /app/run.py

STOPSIGNAL SIGTERM

USER runner
WORKDIR /work
ENTRYPOINT ["python3", "-u", "/app/run.py"]
