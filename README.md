# Ryvion runners

This repository owns the workload-specific container images that `hub-orch` can assign to `node-agent`.

Current scope:

- `embed-runner`
- `transcode-runner`
- `vllm-runner`
- `spatial-stage-runner` (published as `spatial-recon-runner`, `pointcloud-align-runner`, `mesh-optimize-runner`, `scene-render-runner`)

Important deployment note:

- `llm-runner` is still built from `/Users/caspian/Developer/Startup/Ryvion/node-agent` and published as `ghcr.io/ryvion/llm-runner`.
- `vllm-runner` in this repository is the batched large-model container family used by `hub-orch` for the GPU-heavy OpenAI-compatible model tags.

Runner contract:

1. `hub-orch` writes a structured `job.json` into `/work/job.json`.
2. `node-agent` mounts `/work` and runs the assigned container image.
3. The runner must:
   - read `/work/job.json`
   - fetch any `payload_url` it needs
   - write its artifact to `/work/output`
   - write `/work/receipt.json`
   - optionally write `/work/metrics.json`

Compatibility:

- The CI workflow publishes the exact image names `hub-orch` routes today:
  - `ghcr.io/ryvion/embed-runner:0.1.0`
  - `ghcr.io/ryvion/transcode-runner:0.1.0`
  - `ghcr.io/ryvion/vllm-runner:{latest,deepseek-r1-671b,deepseek-v3-671b,llama-3_3-70b,qwen-2_5-72b,mistral-large-2}`
  - `ghcr.io/ryvion/spatial-recon-runner:0.1.0`
  - `ghcr.io/ryvion/pointcloud-align-runner:0.1.0`
  - `ghcr.io/ryvion/mesh-optimize-runner:0.1.0`
  - `ghcr.io/ryvion/scene-render-runner:0.1.0`
- Each image also receives `latest` and commit SHA tags for operator testing.
