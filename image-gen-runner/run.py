"""Ryvion image generation runner.

Reads /work/job.json, generates an image with Stable Diffusion XL Turbo,
and writes /work/output.png, /work/receipt.json, and /work/metrics.json.

IMPORTANT: The model is pre-baked in the Docker image. The container runs
with --network=none so no downloads are possible at runtime.
"""

import hashlib
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

OUTPUT_PATH = Path("/work/output.png")
RECEIPT_PATH = Path("/work/receipt.json")
METRICS_PATH = Path("/work/metrics.json")

MODELS = {
    "sdxl-turbo": "stabilityai/sdxl-turbo",
    "stable-diffusion-xl": "stabilityai/stable-diffusion-xl-base-1.0",
}
DEFAULT_MODEL = "sdxl-turbo"

# SDXL Turbo native resolution — going above 512 risks OOM on 16GB cards
TURBO_MAX_SIZE = 512

_shutdown = False

def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True
    sys.exit(143)

signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)


def load_job(path: Path = Path("/work/job.json")) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(json.dumps({"error": f"failed to read job.json: {e}"}), file=sys.stderr)
        return {}


def fail_receipt(error_msg: str):
    """Write a failure receipt so the node-agent can report it."""
    error_hash = hashlib.sha256(error_msg.encode()).hexdigest()
    RECEIPT_PATH.write_text(json.dumps({
        "output_hash": error_hash,
        "ok": False,
        "error": error_msg,
    }), encoding="utf-8")


def main() -> int:
    job = load_job()
    prompt = str(job.get("prompt") or "").strip()
    if not prompt:
        print("ERROR: prompt is required", file=sys.stderr)
        fail_receipt("prompt is required")
        return 1

    model_name = str(job.get("model") or DEFAULT_MODEL).strip()
    hf_model = MODELS.get(model_name, MODELS[DEFAULT_MODEL])
    width = int(job.get("width") or 512)
    height = int(job.get("height") or 512)
    quality = str(job.get("quality") or "standard").strip()

    # Clamp SDXL Turbo to its native resolution to avoid OOM
    if model_name == "sdxl-turbo":
        width = min(width, TURBO_MAX_SIZE)
        height = min(height, TURBO_MAX_SIZE)
        num_steps = 4 if quality == "standard" else 8
        guidance_scale = 0.0
    else:
        width = min(width, 1024)
        height = min(height, 1024)
        num_steps = 30 if quality == "standard" else 50
        guidance_scale = 7.5

    # Ensure dimensions are multiples of 8 (required by diffusers)
    width = (width // 8) * 8
    height = (height // 8) * 8
    if width < 256:
        width = 256
    if height < 256:
        height = 256

    started = time.time()

    try:
        import torch
        from diffusers import AutoPipelineForText2Image

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        print(json.dumps({
            "event": "loading_model",
            "model": hf_model,
            "device": device,
            "dtype": str(dtype),
        }), file=sys.stderr)

        pipe = AutoPipelineForText2Image.from_pretrained(
            hf_model,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
            local_files_only=True,  # Never try to download — model is pre-baked
        )
        pipe = pipe.to(device)

        if device == "cuda":
            pipe.enable_attention_slicing()

        print(json.dumps({
            "event": "generating",
            "prompt": prompt[:100],
            "width": width,
            "height": height,
            "steps": num_steps,
        }), file=sys.stderr)

        result = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
        )
        image = result.images[0]
        image.save(str(OUTPUT_PATH), format="PNG")

    except Exception as e:
        tb = traceback.format_exc()
        print(json.dumps({"error": str(e), "traceback": tb}), file=sys.stderr)
        fail_receipt(str(e))
        return 1

    duration_ms = int((time.time() - started) * 1000)
    image_bytes = OUTPUT_PATH.read_bytes()
    output_hash = hashlib.sha256(image_bytes).hexdigest()

    RECEIPT_PATH.write_text(json.dumps({
        "output_hash": output_hash,
        "ok": True,
        "model": model_name,
        "hf_model": hf_model,
        "width": width,
        "height": height,
        "steps": num_steps,
        "guidance_scale": guidance_scale,
        "duration_ms": duration_ms,
        "output_bytes": len(image_bytes),
    }), encoding="utf-8")

    METRICS_PATH.write_text(json.dumps({
        "engine": "diffusers",
        "model": model_name,
        "duration_ms": duration_ms,
        "output_bytes": len(image_bytes),
        "output_name": "output.png",
        "width": width,
        "height": height,
    }), encoding="utf-8")

    print(json.dumps({
        "output_hash": output_hash,
        "ok": True,
        "model": model_name,
        "duration_ms": duration_ms,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
