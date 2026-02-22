"""Ryvion LLM inference runner.

Reads /work/job.json, runs inference with llama-cpp-python, writes /work/receipt.json.
Designed to run inside an OCI container on DePIN compute nodes.
"""

import hashlib
import json
import os
import signal
import sys
import time

from llama_cpp import Llama

_shutdown = False

def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True
    print(json.dumps({"event": "sigterm_received"}), file=sys.stderr)
    sys.exit(143)  # 128 + 15 (SIGTERM)

signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)

MODEL_DIR = os.environ.get("RYV_MODEL_DIR", "/models")
DEFAULT_MODEL = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MAX_CONTEXT = int(os.environ.get("RYV_CTX_SIZE", "2048"))
N_THREADS = int(os.environ.get("RYV_THREADS", "4"))
GPU_LAYERS = int(os.environ.get("RYV_GPU_LAYERS", "0"))


def find_model():
    """Find the GGUF model file."""
    path = os.path.join(MODEL_DIR, DEFAULT_MODEL)
    if os.path.isfile(path):
        return path
    # Fall back to any .gguf file in the model directory
    for f in os.listdir(MODEL_DIR):
        if f.endswith(".gguf"):
            return os.path.join(MODEL_DIR, f)
    return None


def load_job(path="/work/job.json"):
    """Load job specification."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(json.dumps({"error": f"failed to read job.json: {e}"}), file=sys.stderr)
        return {}


def run_inference(model_path, job):
    """Run LLM inference and return (response_text, token_count, duration_ms)."""
    messages = job.get("messages", [])
    prompt = job.get("prompt", "")
    max_tokens = job.get("max_tokens", 256)
    temperature = job.get("temperature", 0.7)

    llm = Llama(
        model_path=model_path,
        n_ctx=MAX_CONTEXT,
        n_threads=N_THREADS,
        n_gpu_layers=GPU_LAYERS,
        verbose=False,
    )

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
        result = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        response = result["choices"][0]["text"]
        tokens = result["usage"]["completion_tokens"]

    duration_ms = int((time.time() - start) * 1000)
    return response, tokens, duration_ms


def main():
    job = load_job()
    model_path = find_model()

    if not model_path:
        err = {"error": "no model found", "model_dir": MODEL_DIR}
        print(json.dumps(err), file=sys.stderr)
        with open("/work/receipt.json", "w") as f:
            json.dump({"output_hash": hashlib.sha256(b"error").hexdigest()}, f)
        sys.exit(1)

    try:
        response, tokens, duration_ms = run_inference(model_path, job)
    except Exception as e:
        err = {"error": str(e)}
        print(json.dumps(err), file=sys.stderr)
        with open("/work/receipt.json", "w") as f:
            json.dump({"output_hash": hashlib.sha256(str(e).encode()).hexdigest()}, f)
        sys.exit(1)

    output_hash = hashlib.sha256(response.encode()).hexdigest()

    receipt = {
        "output_hash": output_hash,
        "response": response,
        "model": os.path.splitext(os.path.basename(model_path))[0],
        "tokens_generated": tokens,
        "duration_ms": duration_ms,
    }
    with open("/work/receipt.json", "w") as f:
        json.dump(receipt, f)

    # Print structured output for node-agent log capture
    output = {
        "response": response,
        "tokens": tokens,
        "duration_ms": duration_ms,
        "model": receipt["model"],
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
