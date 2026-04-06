#!/usr/bin/env python3
# Immediate diagnostic output before ANY imports
import sys, os
print("FINETUNE_RUNNER_START", file=sys.stderr, flush=True)
print(f"python={sys.version}", file=sys.stderr, flush=True)
print(f"cwd={os.getcwd()}", file=sys.stderr, flush=True)
print(f"work_files={os.listdir('/work') if os.path.isdir('/work') else 'NO /work'}", file=sys.stderr, flush=True)
print(f"uid={os.getuid()} gid={os.getgid()}", file=sys.stderr, flush=True)

try:
    import json as _json
    job_path = "/work/job.json"
    if os.path.exists(job_path):
        with open(job_path) as _f:
            _job = _json.load(_f)
        print(f"job_task={_job.get('task','?')} base={_job.get('base_model_id','?')}", file=sys.stderr, flush=True)
    else:
        print("NO job.json found!", file=sys.stderr, flush=True)
except Exception as _e:
    print(f"DIAG_ERROR: {_e}", file=sys.stderr, flush=True)

print("IMPORTS_START", file=sys.stderr, flush=True)

try:
    import hashlib
    print("import hashlib ok", file=sys.stderr, flush=True)
    import json
    print("import json ok", file=sys.stderr, flush=True)
    import signal
    print("import signal ok", file=sys.stderr, flush=True)
    import time
    print("import time ok", file=sys.stderr, flush=True)
    import traceback
    print("import traceback ok", file=sys.stderr, flush=True)
except Exception as _e2:
    print(f"STDLIB_IMPORT_FAILED: {_e2}", file=sys.stderr, flush=True)
    sys.exit(1)

# Test if signal handlers work under --cap-drop=ALL
try:
    signal.signal(signal.SIGTERM, lambda s, f: None)
    print("signal.signal ok", file=sys.stderr, flush=True)
except Exception as _e3:
    print(f"SIGNAL_FAILED: {_e3}", file=sys.stderr, flush=True)

print("ALL_MODULE_IMPORTS_OK", file=sys.stderr, flush=True)

"""Ryvion LoRA fine-tuning runner.

Reads /work/job.json, fine-tunes a base model with LoRA using the Unsloth
library, exports the result as GGUF, and writes /work/receipt.json.

Designed to run inside an OCI container on DePIN compute nodes.

Job spec (from /work/job.json):
{
    "task": "finetune",
    "base_model_id": "unsloth/Llama-3.2-3B-Instruct",
    "training_data_url": "https://...",        # URL to JSONL training file
    "training_blob_path": "path/in/r2",        # Alternative: blob path
    "epochs": 3,
    "learning_rate": 2e-5,
    "batch_size": 4,
    "max_seq_length": 2048,
    "lora_rank": 16,
    "lora_alpha": 32,
    "checkpoint_interval": 100,                # Save checkpoint every N steps
    "finetune_job_id": "ft_...",               # For back-linking to hub
    "hub_url": "https://ryvion-hub.fly.dev",   # For checkpoint uploads
}

Training data format (JSONL, each line):
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
  — or —
{"instruction": "...", "input": "...", "output": "..."}
  — or —
{"text": "..."}
"""

import hashlib
import json
import os
import signal
import sys
import time
import traceback

_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True
    print(json.dumps({"event": "sigterm_received"}), file=sys.stderr)
    sys.exit(143)


signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)


def load_job(path="/work/job.json"):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(json.dumps({"error": f"failed to read job.json: {e}"}), file=sys.stderr)
        return {}


def download_training_data(job):
    """Find or download training data."""
    # Always check local files first (node-agent prefetches into /work/)
    for candidate in ["/work/training.jsonl", "/work/data.jsonl", "/work/train.jsonl"]:
        if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
            print(json.dumps({"event": "using_prefetched_training_data", "path": candidate, "size": os.path.getsize(candidate)}), file=sys.stderr)
            return candidate

    # Fallback: download from URL
    url = job.get("training_data_url", "").strip()
    if not url:
        return None

    import urllib.request
    dest = "/work/training_downloaded.jsonl"
    print(json.dumps({"event": "downloading_training_data", "url": url[:80]}), file=sys.stderr)
    try:
        urllib.request.urlretrieve(url, dest)
        print(json.dumps({"event": "training_data_downloaded", "size": os.path.getsize(dest)}), file=sys.stderr)
        return dest
    except Exception as e:
        print(json.dumps({"error": f"failed to download training data: {e}"}), file=sys.stderr)
        return None


def parse_training_data(path):
    """Parse JSONL training data into a list of training examples.

    Supports three formats:
    1. Chat format: {"messages": [{"role": "user", "content": "..."}, ...]}
    2. Alpaca format: {"instruction": "...", "input": "...", "output": "..."}
    3. Raw text: {"text": "..."}
    """
    examples = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(json.dumps({"warning": f"skipping malformed line {line_num}"}), file=sys.stderr)
                continue

            if "messages" in obj:
                # Chat format → convert to text
                parts = []
                for msg in obj["messages"]:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "system":
                        parts.append(f"### System:\n{content}")
                    elif role == "user":
                        parts.append(f"### User:\n{content}")
                    elif role == "assistant":
                        parts.append(f"### Assistant:\n{content}")
                examples.append({"text": "\n\n".join(parts)})
            elif "instruction" in obj:
                # Alpaca format
                instruction = obj.get("instruction", "")
                inp = obj.get("input", "")
                output = obj.get("output", "")
                text = f"### Instruction:\n{instruction}"
                if inp:
                    text += f"\n\n### Input:\n{inp}"
                text += f"\n\n### Response:\n{output}"
                examples.append({"text": text})
            elif "text" in obj:
                examples.append({"text": obj["text"]})
            else:
                print(json.dumps({"warning": f"skipping unknown format at line {line_num}"}), file=sys.stderr)

    return examples


def run_finetune(job, training_path):
    """Run LoRA fine-tuning with Unsloth and return result info."""
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments
    import torch

    # Parameters
    base_model = job.get("base_model_id", "unsloth/Llama-3.2-3B-Instruct")
    epochs = job.get("epochs", 3)
    lr = job.get("learning_rate", 2e-5)
    batch_size = job.get("batch_size", 4)
    max_seq_length = job.get("max_seq_length", 2048)
    lora_rank = job.get("lora_rank", 16)
    lora_alpha = job.get("lora_alpha", 32)
    checkpoint_interval = job.get("checkpoint_interval", 100)

    print(json.dumps({
        "event": "finetune_start",
        "base_model": base_model,
        "epochs": epochs,
        "learning_rate": lr,
        "lora_rank": lora_rank,
        "max_seq_length": max_seq_length,
    }), file=sys.stderr)

    # Load base model with 4-bit quantization for memory efficiency
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )

    # Apply LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Load and prepare training data
    examples = parse_training_data(training_path)
    if not examples:
        raise ValueError("No valid training examples found in data file")

    dataset = Dataset.from_list(examples)
    print(json.dumps({
        "event": "dataset_loaded",
        "examples": len(examples),
    }), file=sys.stderr)

    # Training arguments
    output_dir = "/work/output"
    os.makedirs(output_dir, exist_ok=True)

    max_steps = len(examples) * epochs // batch_size
    if max_steps < 1:
        max_steps = 1

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=max(1, 4 // batch_size),
        learning_rate=lr,
        weight_decay=0.01,
        warmup_steps=min(10, max_steps // 10),
        logging_steps=max(1, max_steps // 20),
        save_steps=checkpoint_interval,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        optim="adamw_8bit",
        seed=42,
        report_to="none",
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=True,
    )

    start_time = time.time()
    train_result = trainer.train()
    duration_ms = int((time.time() - start_time) * 1000)

    # Log training metrics
    metrics = train_result.metrics
    print(json.dumps({
        "event": "training_complete",
        "duration_ms": duration_ms,
        "train_loss": metrics.get("train_loss", 0),
        "train_steps": metrics.get("train_steps", 0),
        "epochs_completed": epochs,
    }), file=sys.stderr)

    # Save merged model + export to GGUF
    result_dir = "/work/result"
    os.makedirs(result_dir, exist_ok=True)

    print(json.dumps({"event": "exporting_gguf"}), file=sys.stderr)
    model.save_pretrained_gguf(
        result_dir,
        tokenizer,
        quantization_method="q4_k_m",
    )

    # Find the resulting GGUF file
    gguf_files = [f for f in os.listdir(result_dir) if f.endswith(".gguf")]
    if not gguf_files:
        raise ValueError("GGUF export produced no output files")

    gguf_path = os.path.join(result_dir, gguf_files[0])
    gguf_size = os.path.getsize(gguf_path)

    # Copy to /work/output for the node-agent artifact collector
    output_path = "/work/output.bin"
    import shutil
    shutil.copy2(gguf_path, output_path)

    print(json.dumps({
        "event": "gguf_exported",
        "file": gguf_files[0],
        "size_bytes": gguf_size,
    }), file=sys.stderr)

    return {
        "duration_ms": duration_ms,
        "train_loss": metrics.get("train_loss", 0),
        "train_steps": metrics.get("train_steps", 0),
        "epochs": epochs,
        "examples": len(examples),
        "gguf_file": gguf_files[0],
        "gguf_size_bytes": gguf_size,
        "lora_rank": lora_rank,
        "base_model": base_model,
        "output_path": output_path,
    }


def main():
    job = load_job()
    if not job:
        sys.exit(1)

    task = job.get("task", "")
    if task != "finetune":
        print(json.dumps({"error": f"unexpected task: {task}"}), file=sys.stderr)
        sys.exit(1)

    # Download training data
    training_path = download_training_data(job)
    if not training_path:
        error_hash = hashlib.sha256(b"no_training_data").hexdigest()
        with open("/work/receipt.json", "w") as f:
            json.dump({"output_hash": error_hash, "error": "no training data found"}, f)
        sys.exit(1)

    try:
        result = run_finetune(job, training_path)
    except Exception as e:
        tb = traceback.format_exc()
        print(json.dumps({"error": str(e), "traceback": tb}), file=sys.stderr)
        error_hash = hashlib.sha256(str(e).encode()).hexdigest()
        with open("/work/receipt.json", "w") as f:
            json.dump({"output_hash": error_hash, "error": str(e)}, f)
        sys.exit(1)

    # Compute output hash from the GGUF file
    output_path = result.get("output_path", "/work/output.bin")
    h = hashlib.sha256()
    with open(output_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    output_hash = h.hexdigest()

    # Write receipt
    receipt = {
        "output_hash": output_hash,
        "duration_ms": result["duration_ms"],
        "train_loss": result["train_loss"],
        "train_steps": result["train_steps"],
        "epochs": result["epochs"],
        "examples": result["examples"],
        "gguf_file": result["gguf_file"],
        "gguf_size_bytes": result["gguf_size_bytes"],
        "lora_rank": result["lora_rank"],
        "base_model": result["base_model"],
        "result_model_path": output_path,
        "finetune_job_id": job.get("finetune_job_id", ""),
    }
    with open("/work/receipt.json", "w") as f:
        json.dump(receipt, f)

    # Write metrics for node-agent
    with open("/work/metrics.json", "w") as f:
        json.dump({
            "output_name": "output.bin",
            "duration_ms": result["duration_ms"],
            "result_model_path": output_path,
        }, f)

    # Structured stdout for log capture
    print(json.dumps({
        "status": "completed",
        "output_hash": output_hash,
        "train_loss": result["train_loss"],
        "epochs": result["epochs"],
        "examples": result["examples"],
        "gguf_size_bytes": result["gguf_size_bytes"],
        "duration_ms": result["duration_ms"],
    }))


if __name__ == "__main__":
    main()
