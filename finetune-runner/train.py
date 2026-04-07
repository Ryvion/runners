#!/usr/bin/env python3
"""Ryvion LoRA fine-tuning runner.

Uses the official HuggingFace SFTTrainer + PEFT LoRA API (2025/2026).
Reference: https://huggingface.co/docs/trl/sft_trainer
"""
import hashlib
import json
import os
import signal
import sys
import time
import traceback

print("FINETUNE_RUNNER v2", file=sys.stderr, flush=True)

_shutdown = False
def _handle_sigterm(s, f):
    global _shutdown
    _shutdown = True
    sys.exit(143)
signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)


def load_job():
    try:
        with open("/work/job.json") as f:
            return json.load(f)
    except Exception as e:
        print(f"job.json error: {e}", file=sys.stderr, flush=True)
        return {}


def find_training_data(job):
    for path in ["/work/training.jsonl", "/work/data.jsonl", "/work/train.jsonl"]:
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            print(json.dumps({"event": "found_data", "path": path, "size": os.path.getsize(path)}), file=sys.stderr, flush=True)
            return path
    url = job.get("training_data_url", "").strip()
    if url:
        import urllib.request
        dest = "/work/training_dl.jsonl"
        try:
            urllib.request.urlretrieve(url, dest)
            return dest
        except Exception as e:
            print(f"download error: {e}", file=sys.stderr, flush=True)
    return None


def parse_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "messages" in obj:
                parts = [f"### {m['role'].title()}:\n{m['content']}" for m in obj["messages"]]
                rows.append({"text": "\n\n".join(parts)})
            elif "instruction" in obj:
                t = f"### Instruction:\n{obj['instruction']}"
                if obj.get("input"):
                    t += f"\n\n### Input:\n{obj['input']}"
                t += f"\n\n### Response:\n{obj.get('output', '')}"
                rows.append({"text": t})
            elif "text" in obj:
                rows.append({"text": obj["text"]})
    return rows


def fail(msg):
    h = hashlib.sha256(msg.encode()).hexdigest()
    with open("/work/receipt.json", "w") as f:
        json.dump({"output_hash": h, "error": msg}, f)
    return 1


def main():
    job = load_job()
    if not job or job.get("task") != "finetune":
        return fail("invalid job")

    data_path = find_training_data(job)
    if not data_path:
        return fail("no training data")

    base_model = job.get("base_model_id", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    epochs = job.get("epochs", 3)
    lr = job.get("learning_rate", 2e-5)
    lora_rank = job.get("lora_rank", 16)

    print(json.dumps({"event": "start", "model": base_model, "epochs": epochs}), file=sys.stderr, flush=True)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig
        from trl import SFTConfig, SFTTrainer
        from datasets import Dataset

        print(json.dumps({"event": "imports_ok", "torch": torch.__version__, "cuda": torch.cuda.is_available()}), file=sys.stderr, flush=True)

        # Parse data
        rows = parse_jsonl(data_path)
        if not rows:
            return fail("no valid examples")
        dataset = Dataset.from_list(rows)
        print(json.dumps({"event": "dataset", "n": len(rows)}), file=sys.stderr, flush=True)

        # 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        # Load model
        print(json.dumps({"event": "loading_model"}), file=sys.stderr, flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config if torch.cuda.is_available() else None,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # LoRA config
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Training config — using SFTConfig (not TrainingArguments)
        output_dir = "/work/output"
        os.makedirs(output_dir, exist_ok=True)

        sft_config = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            learning_rate=lr,
            weight_decay=0.01,
            warmup_steps=5,
            logging_steps=1,
            save_strategy="no",
            fp16=torch.cuda.is_available(),
            optim="adamw_torch",
            seed=42,
            report_to="none",
            max_seq_length=1024,
            packing=False,
            dataset_text_field="text",
        )

        # Train — pass peft_config directly, SFTTrainer handles everything
        print(json.dumps({"event": "training_start"}), file=sys.stderr, flush=True)
        start = time.time()

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=sft_config,
            peft_config=peft_config,
            processing_class=tokenizer,
        )
        result = trainer.train()
        duration_ms = int((time.time() - start) * 1000)
        loss = result.metrics.get("train_loss", 0)

        print(json.dumps({"event": "training_done", "duration_ms": duration_ms, "loss": loss}), file=sys.stderr, flush=True)

        # Save merged model as fp16 safetensors (dequantize from 4-bit so GGUF conversion works)
        print(json.dumps({"event": "saving_merged"}), file=sys.stderr, flush=True)
        merged = trainer.model.merge_and_unload()
        # Dequantize to fp16 — bitsandbytes 4-bit can't be directly converted to GGUF
        merged = merged.to(torch.float16)
        save_dir = "/work/result"
        merged.save_pretrained(save_dir, safe_serialization=True)
        tokenizer.save_pretrained(save_dir)

        # Convert to GGUF for direct use with llama-server
        output_path = None
        output_format = None

        # Try GGUF conversion first
        try:
            import subprocess
            convert_script = "/opt/llama.cpp/convert_hf_to_gguf.py"
            if os.path.exists(convert_script):
                print(json.dumps({"event": "converting_gguf"}), file=sys.stderr, flush=True)
                gguf_path = "/work/output.gguf"
                proc = subprocess.run(
                    ["python3", convert_script, save_dir, "--outtype", "q8_0", "--outfile", gguf_path],
                    capture_output=True, text=True, timeout=300
                )
                if proc.returncode == 0 and os.path.exists(gguf_path) and os.path.getsize(gguf_path) > 0:
                    output_path = gguf_path
                    output_format = "gguf"
                    print(json.dumps({"event": "gguf_exported", "size": os.path.getsize(output_path)}), file=sys.stderr, flush=True)
                else:
                    print(json.dumps({"event": "gguf_failed", "stderr": proc.stderr[-500:]}), file=sys.stderr, flush=True)
        except Exception as conv_err:
            print(json.dumps({"event": "gguf_error", "error": str(conv_err)}), file=sys.stderr, flush=True)

        # Fallback: zip the safetensors
        if output_path is None:
            import shutil
            print(json.dumps({"event": "saving_zip"}), file=sys.stderr, flush=True)
            output_path = "/work/output.zip"
            shutil.make_archive("/work/output", "zip", save_dir)
            output_format = "safetensors_zip"

        output_size = os.path.getsize(output_path)
        output_name = os.path.basename(output_path)
        print(json.dumps({"event": "saved", "size": output_size, "format": output_format, "filename": output_name}), file=sys.stderr, flush=True)

    except Exception as e:
        print(json.dumps({"error": str(e), "tb": traceback.format_exc()}), file=sys.stderr, flush=True)
        return fail(str(e))

    # Hash + receipt
    h = hashlib.sha256()
    with open(output_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)

    with open("/work/receipt.json", "w") as f:
        json.dump({
            "output_hash": h.hexdigest(),
            "duration_ms": duration_ms,
            "train_loss": loss,
            "epochs": epochs,
            "examples": len(rows),
            "base_model": base_model,
            "result_model_path": output_path,
            "finetune_job_id": job.get("finetune_job_id", ""),
        }, f)

    with open("/work/metrics.json", "w") as f:
        json.dump({"output_name": output_name, "duration_ms": duration_ms, "result_model_path": output_path, "format": output_format}, f)

    print(json.dumps({"status": "completed", "hash": h.hexdigest(), "duration_ms": duration_ms, "loss": loss}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
