# Ryvion runners

This repository owns the workload-specific container images that `hub-orch` can assign to `node-agent`.

Current scope:

- `embed-runner`
- `agent-runner`
- `image-gen-runner`
- `llm-runner`
- `transcode-runner`
- `vllm-runner`
- `whisper-runner`
- `spatial-stage-runner` (published as `spatial-recon-runner`, `pointcloud-align-runner`, `mesh-optimize-runner`, `scene-render-runner`)

Runner ownership rule:

- Runner images live in this repository.
- `node-agent` is responsible for runtime execution, not for owning runner image definitions.
- `vllm-runner` remains the batched large-model container family used by `hub-orch` for the GPU-heavy OpenAI-compatible model tags.
- `agent-runner` is the default persistent-agent starter image used by `hub-orch` when a buyer deploys an agent without providing a custom image.

Runner contract:

1. `hub-orch` writes a structured `job.json` into `/work/job.json`.
2. `node-agent` mounts `/work` and runs the assigned container image.
3. The runner must:
   - read `/work/job.json`
   - fetch any `payload_url` it needs
   - write its artifact to `/work/output`
   - write `/work/receipt.json`
   - optionally write `/work/metrics.json`

Spatial pipeline stages now perform real geometry work instead of placeholder manifests:

- `spatial-recon-runner`: OpenCV two-view sparse reconstruction from uploaded image sets
- `pointcloud-align-runner`: coarse ICP-style point-cloud alignment
- `mesh-optimize-runner`: mesh generation/cleanup via convex hull + trimesh processing
- `scene-render-runner`: headless preview rendering to PNG for review workflows

Compatibility:

- The CI workflow publishes the exact image names `hub-orch` routes today:
	- `ghcr.io/ryvion/embed-runner:0.1.0`
	- `ghcr.io/ryvion/agent-runner:0.1.0`
	- `ghcr.io/ryvion/image-gen-runner:0.1.0`
	- `ghcr.io/ryvion/llm-runner:0.1.0`
	- `ghcr.io/ryvion/transcode-runner:0.1.0`
	- `ghcr.io/ryvion/vllm-runner:{latest,deepseek-r1-671b,deepseek-v3-671b,llama-3_3-70b,qwen-2_5-72b,mistral-large-2}`
	- `ghcr.io/ryvion/whisper-runner:0.1.0`
	- `ghcr.io/ryvion/spatial-recon-runner:0.1.0`
	- `ghcr.io/ryvion/pointcloud-align-runner:0.1.0`
	- `ghcr.io/ryvion/mesh-optimize-runner:0.1.0`
  - `ghcr.io/ryvion/scene-render-runner:0.1.0`
- Each image also receives `latest` and commit SHA tags for operator testing.
