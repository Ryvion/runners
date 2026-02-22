FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake curl ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir llama-cpp-python==0.3.8

RUN mkdir -p /models && \
    curl -fSL --retry 3 -o /models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /models /models

RUN useradd -r -s /usr/sbin/nologin runner
COPY run.py /app/run.py

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import sys; sys.exit(0)" || exit 1

STOPSIGNAL SIGTERM

USER runner
WORKDIR /work
ENTRYPOINT ["python", "-u", "/app/run.py"]
