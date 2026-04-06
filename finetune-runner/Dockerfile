FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential cmake curl \
    ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support first
RUN pip3 install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu124

# Install training stack
RUN pip3 install --no-cache-dir \
    "unsloth[cu124-torch250]" datasets trl transformers peft \
    bitsandbytes accelerate sentencepiece protobuf triton

# Verify imports work before proceeding to runtime stage
RUN python3 -c "\
import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available()); \
from unsloth import FastLanguageModel; print('unsloth: ok'); \
from trl import SFTTrainer; print('trl: ok'); \
from datasets import Dataset; print('datasets: ok'); \
print('All imports verified')"

# ---------------------------------------------------------------------------
# Use devel image for runtime too — bitsandbytes/triton need CUDA libs
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip libgomp1 ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.10/dist-packages \
     /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY train.py /app/train.py
WORKDIR /work
ENTRYPOINT ["python3", "-u", "/app/train.py"]
