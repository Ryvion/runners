FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential cmake curl \
    ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.4 support (pinned to avoid backtracking)
RUN pip3 install --no-cache-dir \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install unsloth and training stack with pinned versions
RUN pip3 install --no-cache-dir \
    unsloth \
    datasets==3.2.0 \
    trl==0.12.2 \
    transformers==4.47.1 \
    peft==0.14.0 \
    bitsandbytes==0.45.0 \
    accelerate==1.2.1 \
    sentencepiece \
    protobuf

# Verify imports
RUN python3 -c "\
import torch; print('torch:', torch.__version__); \
from unsloth import FastLanguageModel; print('unsloth: ok'); \
from trl import SFTTrainer; print('trl: ok'); \
print('All imports OK')"

COPY train.py /app/train.py
WORKDIR /work
ENTRYPOINT ["python3", "-u", "/app/train.py"]
