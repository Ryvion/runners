"""Ryvion custom model runner.

Downloads a buyer's custom model from a presigned URL, loads it based on format,
runs inference with the provided input, and writes /work/receipt.json.

Supported formats:
  - GGUF: loaded via llama-cpp-python (CPU or GPU)
  - ONNX: loaded via onnxruntime (CPU or CUDA)
  - PyTorch/SafeTensors: loaded via transformers (auto device)
  - Custom: treated as a Python script with a predict() function

Reads /work/job.json with spec:
{
  "model_url": "https://presigned-url...",
  "model_format": "gguf|onnx|pytorch|safetensors|custom",
  "model_name": "my-model",
  "input": {"prompt": "...", "max_tokens": 256, ...},
  "task": "custom_inference"
}
"""

import hashlib
import json
import os
import signal
import sys
import time
import urllib.request
from pathlib import Path

_shutdown = False

def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True
    sys.exit(143)

signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)

WORK_DIR = Path("/work")
MODEL_DIR = Path("/tmp/model")
RECEIPT_PATH = WORK_DIR / "receipt.json"
METRICS_PATH = WORK_DIR / "metrics.json"
OUTPUT_PATH = WORK_DIR / "output"

MAX_DOWNLOAD_BYTES = 4 * 1024 * 1024 * 1024  # 4GB


def log(event, **kwargs):
    print(json.dumps({"event": event, **kwargs}), file=sys.stderr)


def fail(error, code=1):
    log("error", error=error)
    output_hash = hashlib.sha256(error.encode()).hexdigest()
    RECEIPT_PATH.write_text(json.dumps({"output_hash": output_hash, "error": error}))
    sys.exit(code)


def load_job():
    try:
        return json.loads((WORK_DIR / "job.json").read_text())
    except Exception as e:
        fail(f"failed to read job.json: {e}")


def download_model(url, filename="model"):
    """Download model from presigned URL."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODEL_DIR / filename

    log("downloading_model", url=url[:80] + "...")
    start = time.time()

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=300) as resp:
            size = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    size += len(chunk)
                    if size > MAX_DOWNLOAD_BYTES:
                        fail(f"model exceeds {MAX_DOWNLOAD_BYTES // (1024**3)}GB limit")
    except Exception as e:
        fail(f"model download failed: {e}")

    duration = time.time() - start
    log("model_downloaded", size_mb=round(size / (1024 * 1024), 1), duration_s=round(duration, 1))
    return dest


def run_gguf(model_path, input_data):
    """Run GGUF model inference via llama-cpp-python."""
    from llama_cpp import Llama

    gpu_layers = int(os.environ.get("RYV_GPU_LAYERS", "-1"))  # -1 = all layers on GPU
    n_ctx = int(os.environ.get("RYV_CTX_SIZE", "4096"))

    llm = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_gpu_layers=gpu_layers,
        verbose=False,
    )

    messages = input_data.get("messages", [])
    prompt = input_data.get("prompt", "")
    max_tokens = input_data.get("max_tokens", 256)
    temperature = input_data.get("temperature", 0.7)

    start = time.time()

    if messages:
        result = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        response = result["choices"][0]["message"]["content"]
        tokens = result["usage"]["completion_tokens"]
    else:
        if not prompt:
            prompt = "Hello"
        result = llm(prompt, max_tokens=max_tokens, temperature=temperature)
        response = result["choices"][0]["text"]
        tokens = result["usage"]["completion_tokens"]

    duration_ms = int((time.time() - start) * 1000)
    return response, tokens, duration_ms


def run_onnx(model_path, input_data):
    """Run ONNX model inference via onnxruntime."""
    import numpy as np
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=providers)

    start = time.time()

    # Build input from job data
    inputs = {}
    for inp in session.get_inputs():
        name = inp.name
        if name in input_data:
            val = input_data[name]
            if isinstance(val, list):
                inputs[name] = np.array(val, dtype=np.float32)
            else:
                inputs[name] = np.array([val], dtype=np.float32)
        else:
            # Create dummy input matching expected shape
            shape = [d if isinstance(d, int) else 1 for d in inp.shape]
            inputs[name] = np.zeros(shape, dtype=np.float32)

    outputs = session.run(None, inputs)
    duration_ms = int((time.time() - start) * 1000)

    # Format output
    result = {}
    for i, out in enumerate(session.get_outputs()):
        result[out.name] = outputs[i].tolist()

    response = json.dumps(result)
    return response, 0, duration_ms


def run_transformers(model_path, input_data, format_type):
    """Run PyTorch/SafeTensors model inference via transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    log("loading_model", device=device, format=format_type)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path.parent), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path.parent),
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    prompt = input_data.get("prompt", "")
    messages = input_data.get("messages", [])
    max_tokens = input_data.get("max_tokens", 256)
    temperature = input_data.get("temperature", 0.7)

    if messages and hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif messages:
        text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
    else:
        text = prompt or "Hello"

    start = time.time()
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 0.01),
            do_sample=temperature > 0,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    duration_ms = int((time.time() - start) * 1000)

    return response, int(tokens), duration_ms


def main():
    job = load_job()
    model_url = job.get("model_url", "")
    model_format = job.get("model_format", "").lower()
    model_name = job.get("model_name", "custom-model")
    input_data = job.get("input", {})

    if not model_url:
        fail("no model_url in job spec")

    # Determine filename from format
    ext_map = {"gguf": ".gguf", "onnx": ".onnx", "pytorch": ".bin", "safetensors": ".safetensors"}
    filename = f"model{ext_map.get(model_format, '.bin')}"

    model_path = download_model(model_url, filename)

    log("running_inference", model=model_name, format=model_format)

    try:
        if model_format == "gguf":
            response, tokens, duration_ms = run_gguf(model_path, input_data)
        elif model_format == "onnx":
            response, tokens, duration_ms = run_onnx(model_path, input_data)
        elif model_format in ("pytorch", "safetensors"):
            response, tokens, duration_ms = run_transformers(model_path, input_data, model_format)
        else:
            # Generic: try ONNX first, fall back to GGUF
            try:
                response, tokens, duration_ms = run_onnx(model_path, input_data)
            except Exception:
                response, tokens, duration_ms = run_gguf(model_path, input_data)
    except Exception as e:
        fail(f"inference failed: {e}")
        return  # unreachable but makes type checker happy

    output_hash = hashlib.sha256(response.encode() if isinstance(response, str) else json.dumps(response).encode()).hexdigest()

    # Write output
    OUTPUT_PATH.write_text(response if isinstance(response, str) else json.dumps(response))

    # Write receipt
    receipt = {
        "output_hash": output_hash,
        "model": model_name,
        "model_format": model_format,
        "tokens_generated": tokens,
        "duration_ms": duration_ms,
    }
    RECEIPT_PATH.write_text(json.dumps(receipt))

    # Write metrics
    metrics = {
        "output_name": "output",
        "model": model_name,
        "format": model_format,
        "tokens": tokens,
        "duration_ms": duration_ms,
        "output_bytes": OUTPUT_PATH.stat().st_size,
    }
    METRICS_PATH.write_text(json.dumps(metrics))

    log("inference_complete", model=model_name, tokens=tokens, duration_ms=duration_ms)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
