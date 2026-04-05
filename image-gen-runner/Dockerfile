FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install --no-cache-dir \
    diffusers[torch] transformers accelerate safetensors pillow huggingface_hub

# Pre-download SDXL Turbo model into the image (avoids runtime network dependency)
RUN python3 -c "\
from diffusers import AutoPipelineForText2Image; \
import torch; \
AutoPipelineForText2Image.from_pretrained('stabilityai/sdxl-turbo', torch_dtype=torch.float16, variant='fp16')"

# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.10/dist-packages \
     /usr/local/lib/python3.10/dist-packages
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

RUN useradd -r -s /usr/sbin/nologin runner && \
    mkdir -p /work && chown runner:runner /work && \
    cp -r /root/.cache/huggingface /home/runner/.cache/huggingface 2>/dev/null || true && \
    chown -R runner:runner /home/runner/.cache 2>/dev/null || true

COPY run.py /app/run.py

USER runner
ENV HF_HOME=/home/runner/.cache/huggingface
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
WORKDIR /work
ENTRYPOINT ["python3", "-u", "/app/run.py"]
