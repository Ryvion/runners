FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install --no-cache-dir \
    diffusers[torch] transformers accelerate safetensors pillow huggingface_hub

# Pre-download SDXL Turbo model — cache-bust v3
RUN python3 -c "\
from diffusers import AutoPipelineForText2Image; \
import torch, os; \
p = AutoPipelineForText2Image.from_pretrained('stabilityai/sdxl-turbo', torch_dtype=torch.float16, variant='fp16'); \
cache = os.path.expanduser('~/.cache/huggingface/hub'); \
total = sum(os.path.getsize(os.path.join(d,f)) for d,_,files in os.walk(cache) for f in files); \
print(f'SDXL Turbo cached: {total/1024/1024:.0f} MB')"

# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.10/dist-packages \
     /usr/local/lib/python3.10/dist-packages
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Run as root to avoid cache permission issues. Security is enforced by
# --cap-drop=ALL, --network=none, and --security-opt=no-new-privileges
# at the docker run level (in node-agent/runner/oci.go).
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

COPY run.py /app/run.py
WORKDIR /work
ENTRYPOINT ["python3", "-u", "/app/run.py"]
