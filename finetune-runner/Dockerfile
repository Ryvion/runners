FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential cmake curl \
    ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.4
RUN pip3 install --no-cache-dir \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install unsloth (brings compatible transformers/trl/peft)
RUN pip3 install --no-cache-dir unsloth

# Install remaining deps
RUN pip3 install --no-cache-dir \
    datasets bitsandbytes accelerate sentencepiece protobuf

# Verify non-GPU imports work (unsloth needs GPU at import, skip it here)
RUN python3 -c "\
import torch; print('torch:', torch.__version__); \
from trl import SFTTrainer; print('trl: ok'); \
from datasets import Dataset; print('datasets: ok'); \
from peft import LoraConfig; print('peft: ok'); \
print('ALL IMPORTS OK (unsloth verified at runtime with GPU)')"

COPY train.py /app/train.py
WORKDIR /work
ENTRYPOINT ["python3", "-u", "/app/train.py"]
