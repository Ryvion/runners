#!/usr/bin/env python3
"""Ryvion LoRA fine-tuning runner (transformers + peft + trl).

Reads /work/job.json, fine-tunes a base model with LoRA,
saves the merged model, and writes /work/receipt.json.
"""
import hashlib
import json
import os
import signal
import sys
import time
import traceback

print("FINETUNE_RUNNER_START", file=sys.stderr, flush=True)

_shutdown = False
def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True
    sys.exit(143)

signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)


def load_job(path="/work/job.json"):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(json.dumps({"error": f"job.json: {e}"}), file=sys.stderr, flush=True)
        return {}


def find_training_data(job):
    """Find prefetched training data or download it."""
    for candidate in ["/work/training.jsonl", "/work/data.jsonl", "/work/train.jsonl"]:
        if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
            print(json.dumps({"event": "found_training_data", "path": candidate, "size": os.path.getsize(candidate)}), file=sys.stderr, flush=True)
            return candidate

    url = job.get("training_data_url", "").strip()
    if not url:
        return None

    import urllib.request
    dest = "/work/training_dl.jsonl"
    try:
        urllib.request.urlretrieve(url, dest)
        return dest
    except Exception as e:
        print(json.dumps({"error": f"download failed: {e}"}), file=sys.stderr, flush=True)
        return None


def parse_training_data(path):
    """Parse JSONL into text examples."""
    examples = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "messages" in obj:
                parts = []
                for msg in obj["messages"]:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    parts.append(f"### {role.title()}:\n{content}")
                examples.append({"text": "\n\n".join(parts)})
            elif "instruction" in obj:
                text = f"### Instruction:\n{obj['instruction']}"
                if obj.get("input"):
                    text += f"\n\n### Input:\n{obj['input']}"
                text += f"\n\n### Response:\n{obj.get('output', '')}"
                examples.append({"text": text})
            elif "text" in obj:
                examples.append({"text": obj["text"]})
    return examples


def fail_receipt(msg):
    h = hashlib.sha256(msg.encode()).hexdigest()
    with open("/work/receipt.json", "w") as f:
        json.dump({"output_hash": h, "error": msg}, f)


def main():
    job = load_job()
    if not job or job.get("task") != "finetune":
        fail_receipt("invalid job spec")
        return 1

    training_path = find_training_data(job)
    if not training_path:
        fail_receipt("no training data")
        return 1

    base_model = job.get("base_model_id", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    epochs = job.get("epochs", 3)
    lr = job.get("learning_rate", 2e-5)
    batch_size = job.get("batch_size", 4)
    lora_rank = job.get("lora_rank", 16)
    lora_alpha = job.get("lora_alpha", 32)

    print(json.dumps({"event": "starting", "model": base_model, "epochs": epochs, "lr": lr}), file=sys.stderr, flush=True)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer
        from datasets import Dataset

        print(json.dumps({"event": "imports_ok", "torch": torch.__version__, "cuda": torch.cuda.is_available()}), file=sys.stderr, flush=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model with 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        print(json.dumps({"event": "loading_model", "model": base_model}), file=sys.stderr, flush=True)
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config if device == "cuda" else None,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        )

        if device == "cuda":
            model = prepare_model_for_kbit_training(model)

        # Apply LoRA
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        print(json.dumps({"event": "lora_applied", "trainable_params": model.print_trainable_parameters()}), file=sys.stderr, flush=True)

        # Load training data
        examples = parse_training_data(training_path)
        if not examples:
            fail_receipt("no valid training examples")
            return 1

        dataset = Dataset.from_list(examples)
        print(json.dumps({"event": "dataset_loaded", "examples": len(examples)}), file=sys.stderr, flush=True)

        # Train
        output_dir = "/work/output"
        os.makedirs(output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=max(1, 4 // batch_size),
            learning_rate=lr,
            weight_decay=0.01,
            warmup_steps=5,
            logging_steps=1,
            save_strategy="no",
            fp16=device == "cuda",
            optim="adamw_torch",
            seed=42,
            report_to="none",
        )

        # Format dataset for SFTTrainer (trl 0.13+: no dataset_text_field, use formatting_func)
        def formatting_func(example):
            return example["text"]

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=training_args,
            formatting_func=formatting_func,
            max_seq_length=1024,
            packing=False,
        )

        start_time = time.time()
        print(json.dumps({"event": "training_start"}), file=sys.stderr, flush=True)
        result = trainer.train()
        duration_ms = int((time.time() - start_time) * 1000)

        loss = result.metrics.get("train_loss", 0)
        print(json.dumps({"event": "training_done", "duration_ms": duration_ms, "loss": loss}), file=sys.stderr, flush=True)

        # Save merged model
        print(json.dumps({"event": "saving_model"}), file=sys.stderr, flush=True)
        merged = model.merge_and_unload()
        save_dir = "/work/result"
        merged.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        # Create output archive
        import shutil
        output_path = "/work/output.bin"
        shutil.make_archive("/work/result_archive", "zip", save_dir)
        shutil.move("/work/result_archive.zip", output_path)

        output_size = os.path.getsize(output_path)
        print(json.dumps({"event": "model_saved", "size_bytes": output_size}), file=sys.stderr, flush=True)

    except Exception as e:
        tb = traceback.format_exc()
        print(json.dumps({"error": str(e), "traceback": tb}), file=sys.stderr, flush=True)
        fail_receipt(str(e))
        return 1

    # Hash output
    h = hashlib.sha256()
    with open(output_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    output_hash = h.hexdigest()

    # Write receipt
    with open("/work/receipt.json", "w") as f:
        json.dump({
            "output_hash": output_hash,
            "duration_ms": duration_ms,
            "train_loss": loss,
            "epochs": epochs,
            "examples": len(examples),
            "base_model": base_model,
            "result_model_path": output_path,
            "finetune_job_id": job.get("finetune_job_id", ""),
        }, f)

    with open("/work/metrics.json", "w") as f:
        json.dump({"output_name": "output.bin", "duration_ms": duration_ms, "result_model_path": output_path}, f)

    print(json.dumps({"status": "completed", "output_hash": output_hash, "duration_ms": duration_ms, "loss": loss}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
