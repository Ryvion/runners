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

# Patch setup_env.py: split configure and build so we can fix source between them.
# Step 1: Run setup_env to download model, generate kernels, configure cmake (but skip compile).
RUN sed -i 's/run_command(\["cmake", "--build"/# PATCHED: skip build\n    # run_command(["cmake", "--build"/' setup_env.py && \
    /opt/venv/bin/python setup_env.py -md microsoft/BitNet-b1.58-2B-4T -q i2_s || true

# Step 2: Patch const-correctness errors in generated source.
# Upstream bug: assigns const int8_t* to int8_t* in AVX2 kernels.
RUN find . -name "*.cpp" -exec sed -i \
    -e 's/int8_t \* y_col = y + col/const int8_t * y_col = y + col/g' \
    -e 's/int8_t \*scales = /const int8_t *scales = /g' \
    {} + 2>/dev/null || true

# Step 3: Build manually.
RUN cmake --build build --config Release || \
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
