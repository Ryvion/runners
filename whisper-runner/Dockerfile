FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir \
    torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 && \
    python3 -m pip install --no-cache-dir openai-whisper==20250625

# Pre-download whisper models into cache (avoids runtime network dependency)
RUN python3 -c "import whisper; whisper.load_model('base'); whisper.load_model('small')"

# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.10/dist-packages \
     /usr/local/lib/python3.10/dist-packages
COPY --from=builder /root/.cache /root/.cache

RUN useradd -r -s /usr/sbin/nologin runner && \
    mkdir -p /work && chown runner:runner /work && \
    cp -r /root/.cache /home/runner/.cache 2>/dev/null || true && \
    chown -R runner:runner /home/runner/.cache 2>/dev/null || true

COPY run.py /app/run.py

USER runner
ENV XDG_CACHE_HOME=/home/runner/.cache
WORKDIR /work
ENTRYPOINT ["python3", "-u", "/app/run.py"]
