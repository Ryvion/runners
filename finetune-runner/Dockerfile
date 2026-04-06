FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential cmake curl \
    ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

# PyTorch with CUDA 12.4
RUN pip3 install --no-cache-dir \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Standard LoRA training stack — NO unsloth (torchao incompatible with stable torch)
RUN pip3 install --no-cache-dir \
    transformers==4.47.1 \
    peft==0.14.0 \
    trl==0.13.0 \
    datasets==3.2.0 \
    bitsandbytes==0.45.0 \
    accelerate==1.2.1 \
    sentencepiece protobuf

# Verify imports work (no GPU needed for these)
RUN python3 -c "\
import torch; print('torch', torch.__version__); \
from transformers import AutoModelForCausalLM, AutoTokenizer; print('transformers ok'); \
from peft import LoraConfig, get_peft_model; print('peft ok'); \
from trl import SFTTrainer; print('trl ok'); \
from datasets import Dataset; print('datasets ok'); \
print('ALL IMPORTS OK')"

COPY train.py /app/train.py
WORKDIR /work
ENTRYPOINT ["python3", "-u", "/app/train.py"]
