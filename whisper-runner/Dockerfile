FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install --no-cache-dir \
    openai-whisper

COPY run.py /app/run.py
WORKDIR /work
ENTRYPOINT ["python3", "-u", "/app/run.py"]
