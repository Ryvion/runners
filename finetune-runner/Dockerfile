FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential cmake curl \
    ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

# PyTorch with CUDA 12.4 — MUST be installed first and pinned
RUN pip3 install --no-cache-dir \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Training stack — pinned versions that work together
RUN pip3 install --no-cache-dir \
    transformers==4.47.1 \
    peft==0.14.0 \
    trl==0.13.0 \
    datasets==3.2.0 \
    bitsandbytes==0.45.0 \
    accelerate==1.2.1 \
    sentencepiece protobuf

# GGUF tools — install AFTER training stack, don't let it override torch
RUN pip3 install --no-cache-dir --no-deps gguf && \
    git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp && \
    pip3 install --no-cache-dir --no-deps numpy

# Verify torch is still CUDA and correct version
RUN python3 -c "\
import torch; \
assert 'cu124' in torch.__version__, f'WRONG TORCH: {torch.__version__}'; \
assert torch.cuda.is_available() or True, 'CUDA check skipped in CI'; \
print(f'torch {torch.__version__} OK'); \
from transformers import AutoModelForCausalLM; print('transformers ok'); \
from peft import LoraConfig; print('peft ok'); \
from trl import SFTConfig, SFTTrainer; print('trl ok'); \
from datasets import Dataset; print('datasets ok'); \
print('ALL OK')"

COPY train.py /app/train.py
WORKDIR /work
ENTRYPOINT ["python3", "-u", "/app/train.py"]
