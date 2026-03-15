"""Ryvion image generation runner.

Reads /work/job.json, generates an image with Stable Diffusion XL Turbo,
and writes /work/output.png, /work/receipt.json, and /work/metrics.json.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

OUTPUT_PATH = Path("/work/output.png")
RECEIPT_PATH = Path("/work/receipt.json")
METRICS_PATH = Path("/work/metrics.json")

# Model map — model name to HuggingFace ID
MODELS = {
    "sdxl-turbo": "stabilityai/sdxl-turbo",
    "stable-diffusion-xl": "stabilityai/stable-diffusion-xl-base-1.0",
}

DEFAULT_MODEL = "sdxl-turbo"


def load_job(path: Path = Path("/work/job.json")) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    job = load_job()
    prompt = str(job.get("prompt") or "").strip()
    if not prompt:
        print("ERROR: prompt is required", file=sys.stderr)
        return 1

    model_name = str(job.get("model") or DEFAULT_MODEL).strip()
    hf_model = MODELS.get(model_name, MODELS[DEFAULT_MODEL])
    width = int(job.get("width") or 1024)
    height = int(job.get("height") or 1024)
    quality = str(job.get("quality") or "standard").strip()

    # Inference steps based on model and quality
    if model_name == "sdxl-turbo":
        num_steps = 4 if quality == "standard" else 8
        guidance_scale = 0.0  # Turbo uses no CFG
    else:
        num_steps = 30 if quality == "standard" else 50
        guidance_scale = 7.5

    started = time.time()

    # Lazy import to fail fast on missing deps
    import torch
    from diffusers import AutoPipelineForText2Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading {hf_model} on {device} ({dtype})")
    pipe = AutoPipelineForText2Image.from_pretrained(
        hf_model,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )
    pipe = pipe.to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()

    print(f"Generating: {prompt!r} ({width}x{height}, steps={num_steps})")
    result = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
    )
    image = result.images[0]
    image.save(str(OUTPUT_PATH), format="PNG")

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
