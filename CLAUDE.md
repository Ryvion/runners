# Runners

Container images for GPU workloads. Built by GitHub Actions, pushed to `ghcr.io/ryvion/*`.

## Build & CI

`.github/workflows/build.yml` — matrix build for all runners on push to main or version tags. Pushes `:latest`, `:SHA`, and version tags.

No local test suite. Verify with `docker build` and `pip list` (not Python imports — GPU imports fail without GPU).

## Container Contract

All containers:
1. Read `/work/job.json` (written by node-agent)
2. Write `/work/receipt.json` (output hash + metadata)
3. Write `/work/metrics.json` (must include `output_name` field so node-agent finds the artifact)
4. Write output artifact to `/work/`

## Network Policy

- ALL containers run with `--network=none` (no internet access)
- EXCEPT `finetune-runner` which gets `--network=bridge` (needs HuggingFace downloads)
- Node-agent prefetches input files (`training_data_url`, `audio_url`, `model_url`, `payload_url`, `input_url`) into `/work/` before container start

## Runners

### finetune-runner
LoRA fine-tuning with QLoRA 4-bit quantization. CUDA 12.4 base image.

Stack: torch 2.5.1, transformers 4.47.1, peft 0.14.0, trl 0.13.0, bitsandbytes 0.45.0, accelerate 1.2.1.

Flow: load base model in 4-bit → train with SFTTrainer + LoRA → save adapter → reload base in fp16 on CPU → merge adapter → convert to GGUF via llama.cpp `convert_hf_to_gguf.py`. Falls back to safetensors zip if GGUF conversion fails.

Training API: `SFTConfig` (not TrainingArguments), `peft_config` param on SFTTrainer, `processing_class` (not tokenizer), `dataset_text_field` in SFTConfig (not SFTTrainer constructor).

Has `Dockerfile.rocm` variant for AMD GPUs.

### image-gen-runner
SDXL Turbo pre-baked in Docker image. `TRANSFORMERS_OFFLINE=1`. Max 512x512 for Turbo (OOM risk above). Outputs `/work/output.png`.

### whisper-runner
Whisper base+small models pre-baked. Looks for `/work/input_audio` (prefetched by node-agent from `audio_url`). Outputs `/work/output.json`.

### model-runner
Custom model inference. Supports GGUF (llama-cpp-python), ONNX (onnxruntime), PyTorch/SafeTensors (transformers), and custom Python scripts. Checks `/work/model.bin` (prefetched by node-agent from `model_url`) before attempting download.

### llm-runner
TinyLlama 1.1B pre-baked. Uses llama-cpp-python. For non-streaming inference (containerized). Streaming inference uses native llama-server on the node instead.

### embed-runner
Embedding generation. Pushed to `ghcr.io/ryvion/embed-runner`.

### transcode-runner
Media transcoding. Multi-arch: `linux/amd64,linux/arm64`.

### vllm-runner
vLLM-based inference for large models. Tagged per model: `:deepseek-r1-671b`, `:deepseek-v3-671b`, `:llama-3_3-70b`, `:qwen-2_5-72b`, `:mistral-large-2`.

### spatial-stage-runner
Spatial reconstruction pipeline. Build arg `STAGE_KIND` selects the stage.

## Common Gotchas

- NEVER install packages that override pinned torch version (use `--no-deps` for gguf)
- unsloth is incompatible with stable PyTorch (torchao issues) — use standard transformers+peft+trl instead (unsloth is in requirements.txt but not used in train.py)
- bitsandbytes 4-bit models can't be `.to(fp16)` — must save adapter, reload base in fp16 on CPU, then merge
- GPU imports fail in CI (no GPU) — verify with `pip list`, not Python imports
- SFTTrainer API: use `SFTConfig` (not `TrainingArguments`), `peft_config` param, `processing_class` (not `tokenizer`), `dataset_text_field` in SFTConfig (not SFTTrainer constructor)
- All runners use signal handlers for SIGTERM (exit 143) — containers get graceful shutdown
- `metrics.json` must have `output_name` field or node-agent won't find the artifact
