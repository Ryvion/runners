FROM golang:1.24-bookworm AS builder

WORKDIR /src
COPY go.mod .
COPY main.go .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags='-s -w' -o /out/embed-runner .

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir sentence-transformers==3.0.1
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.save("/opt/models/all-MiniLM-L6-v2")
PY
ENV RYV_EMBED_MODEL=/opt/models/all-MiniLM-L6-v2 \
    TRANSFORMERS_OFFLINE=1

COPY --from=builder /out/embed-runner /usr/local/bin/embed-runner
COPY llama-embed.py /app/llama-embed.py

RUN printf '%s\n' '#!/bin/sh' 'exec python /app/llama-embed.py "$@"' > /usr/local/bin/llama-embed \
    && chmod +x /usr/local/bin/llama-embed \
    && ln -sf /usr/local/bin/llama-embed /usr/local/bin/llama-cli \
    && ln -sf /usr/local/bin/llama-embed /usr/local/bin/llama-quantize \
    && useradd -r -s /usr/sbin/nologin runner

USER runner
WORKDIR /work
ENTRYPOINT ["/usr/local/bin/embed-runner"]
