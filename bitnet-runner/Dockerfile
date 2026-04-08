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
RUN git clone --recursive --depth 1 https://github.com/microsoft/BitNet.git /opt/bitnet
WORKDIR /opt/bitnet

# Install Python deps in venv
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Patch setup_env.py to add -fpermissive to cmake flags
# (fixes const int8_t* -> int8_t* assignment error in ggml-bitnet-mad.cpp with clang-18)
RUN sed -i 's/"-DCMAKE_CXX_COMPILER=clang++"/"-DCMAKE_CXX_COMPILER=clang++", "-DCMAKE_CXX_FLAGS=-fpermissive"/g' setup_env.py

# setup_env.py: downloads model, generates kernels, compiles llama-cli
RUN /opt/venv/bin/python setup_env.py -md microsoft/BitNet-b1.58-2B-4T -q i2_s || \
    (cat logs/compile.log 2>/dev/null; exit 1)

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
