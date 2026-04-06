FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential cmake curl \
    ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.4
RUN pip3 install --no-cache-dir \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install unsloth first (it pins compatible versions of transformers/trl/peft)
RUN pip3 install --no-cache-dir unsloth

# Install remaining deps (versions resolved by unsloth's constraints)
RUN pip3 install --no-cache-dir \
    datasets bitsandbytes accelerate sentencepiece protobuf

# Verify imports
RUN python3 -c "\
import torch; print('torch:', torch.__version__); \
from unsloth import FastLanguageModel; print('unsloth: ok'); \
from trl import SFTTrainer; print('trl: ok'); \
from datasets import Dataset; print('datasets: ok'); \
print('ALL IMPORTS OK')"

COPY train.py /app/train.py
WORKDIR /work
ENTRYPOINT ["python3", "-u", "/app/train.py"]
