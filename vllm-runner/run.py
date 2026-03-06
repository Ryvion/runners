"""Ryvion vLLM runner.

Reads /work/job.json, runs a single offline generation with vLLM, and writes
/work/output, /work/receipt.json, and /work/metrics.json.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

from vllm import LLM, SamplingParams

MODEL_NAME = os.environ.get("RYV_VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
TENSOR_PARALLEL_SIZE = int(os.environ.get("RYV_TENSOR_PARALLEL_SIZE", "1"))
MAX_MODEL_LEN = int(os.environ.get("RYV_MAX_MODEL_LEN", "8192"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("RYV_GPU_MEMORY_UTILIZATION", "0.92"))
OUTPUT_PATH = Path("/work/output")
RECEIPT_PATH = Path("/work/receipt.json")
METRICS_PATH = Path("/work/metrics.json")


def load_job(path: Path = Path("/work/job.json")) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_prompt(job: dict) -> str:
    messages = job.get("messages") or []
    if messages:
        return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
    prompt = str(job.get("prompt") or "").strip()
    if prompt:
        return prompt
    payload_url = str(job.get("payload_url") or "").strip()
    if payload_url:
        return f"Summarize the payload at {payload_url}."
    return "Hello from Ryvion vLLM runner."


def main() -> int:
    job = load_job()
    prompt = build_prompt(job)
    max_tokens = int(job.get("max_tokens") or 512)
    temperature = float(job.get("temperature") or 0.7)

    started = time.time()
    llm = LLM(
        model=str(job.get("model") or MODEL_NAME),
        tensor_parallel_size=max(1, int(job.get("tensor_parallel_size") or TENSOR_PARALLEL_SIZE)),
        max_model_len=max(1024, int(job.get("max_model_len") or MAX_MODEL_LEN)),
        gpu_memory_utilization=float(job.get("gpu_memory_utilization") or GPU_MEMORY_UTILIZATION),
        trust_remote_code=True,
    )
    outputs = llm.generate([prompt], SamplingParams(max_tokens=max_tokens, temperature=temperature))
    text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    duration_ms = int((time.time() - started) * 1000)

    OUTPUT_PATH.write_text(text, encoding="utf-8")
    output_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    RECEIPT_PATH.write_text(json.dumps({
        "output_hash": output_hash,
        "ok": True,
        "model": str(job.get("model") or MODEL_NAME),
        "tokens_generated": len(text.split()),
        "duration_ms": duration_ms,
    }), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps({
        "engine": "vllm",
        "duration_ms": duration_ms,
        "output_bytes": len(text.encode("utf-8")),
    }), encoding="utf-8")
    print(json.dumps({"output_hash": output_hash, "ok": True, "model": str(job.get("model") or MODEL_NAME)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
