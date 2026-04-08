"""Ryvion BitNet 1-bit LLM inference runner.

Reads /work/job.json, runs inference via bitnet.cpp's llama-cli (CPU-only),
writes /work/receipt.json + /work/metrics.json.
Designed to run inside an OCI container on DePIN compute nodes.
"""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time

_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True
    print(json.dumps({"event": "sigterm_received"}), file=sys.stderr)
    sys.exit(143)  # 128 + 15 (SIGTERM)


signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)

BITNET_DIR = "/opt/bitnet"
LLAMA_CLI = os.path.join(BITNET_DIR, "build", "bin", "llama-cli")
MODEL_PATH = os.path.join(BITNET_DIR, "models", "BitNet-b1.58-2B-4T", "ggml-model-i2_s.gguf")
N_THREADS = int(os.environ.get("RYV_THREADS", str(os.cpu_count() or 4)))


def load_job(path="/work/job.json"):
    """Load job specification."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(json.dumps({"error": f"failed to read job.json: {e}"}), file=sys.stderr)
        return {}


def build_prompt(job):
    """Build a text prompt from job spec (messages list or raw prompt)."""
    messages = job.get("messages", [])
    prompt = job.get("prompt", "")

    if messages and not prompt:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"<|system|>\n{content}<|end|>")
            elif role == "user":
                parts.append(f"<|user|>\n{content}<|end|>")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}<|end|>")
        parts.append("<|assistant|>")
        prompt = "\n".join(parts)

    return prompt or "Hello"


def run_inference(prompt, job):
    """Run inference via llama-cli and return (output_text, duration_ms)."""
    max_tokens = job.get("max_tokens", 512)
    temperature = job.get("temperature", 0.8)

    cmd = [
        LLAMA_CLI,
        "-m", MODEL_PATH,
        "-p", prompt,
        "-n", str(max_tokens),
        "-t", str(N_THREADS),
        "-ngl", "0",
        "-c", "2048",
        "--temp", str(temperature),
        "-b", "1",
        "--no-display-prompt",
    ]

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    duration_ms = int((time.time() - start) * 1000)

    if result.returncode != 0:
        raise RuntimeError(f"llama-cli exited {result.returncode}: {result.stderr[:500]}")

    return result.stdout.strip(), duration_ms


def main():
    job = load_job()

    if not os.path.isfile(LLAMA_CLI):
        err = {"error": "llama-cli binary not found", "path": LLAMA_CLI}
        print(json.dumps(err), file=sys.stderr)
        with open("/work/receipt.json", "w") as f:
            json.dump({"output_hash": hashlib.sha256(b"error").hexdigest()}, f)
        sys.exit(1)

    if not os.path.isfile(MODEL_PATH):
        err = {"error": "model file not found", "path": MODEL_PATH}
        print(json.dumps(err), file=sys.stderr)
        with open("/work/receipt.json", "w") as f:
            json.dump({"output_hash": hashlib.sha256(b"error").hexdigest()}, f)
        sys.exit(1)

    prompt = build_prompt(job)

    try:
        response, duration_ms = run_inference(prompt, job)
    except Exception as e:
        err = {"error": str(e)}
        print(json.dumps(err), file=sys.stderr)
        with open("/work/receipt.json", "w") as f:
            json.dump({"output_hash": hashlib.sha256(str(e).encode()).hexdigest()}, f)
        sys.exit(1)

    # Write output artifact
    with open("/work/output.txt", "w") as f:
        f.write(response)

    output_hash = hashlib.sha256(response.encode()).hexdigest()

    # Write receipt
    receipt = {
        "output_hash": output_hash,
        "output_name": "output.txt",
    }
    with open("/work/receipt.json", "w") as f:
        json.dump(receipt, f)

    # Write metrics (output_name required by node-agent)
    metrics = {
        "output_name": "output.txt",
        "duration_ms": duration_ms,
        "model": "BitNet-b1.58-2B-4T",
        "quantization": "i2_s",
        "cpu_threads": N_THREADS,
    }
    with open("/work/metrics.json", "w") as f:
        json.dump(metrics, f)

    # Structured output for node-agent log capture
    output = {
        "response": response[:500],
        "duration_ms": duration_ms,
        "model": "BitNet-b1.58-2B-4T",
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
