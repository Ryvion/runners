# Ryvion runners

This repository owns the workload-specific container images that `hub-orch` can assign to `node-agent`.

Current scope:

- `embed-runner`
- `transcode-runner`

Important deployment note:

- `llm-runner` is still built from `/Users/caspian/Developer/Startup/Ryvion/node-agent` and published as `ghcr.io/ryvion/llm-runner`.
- This repository intentionally does not publish `llm-runner` to avoid tag collisions until the LLM image is migrated here.

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

- The CI workflow publishes `ghcr.io/ryvion/embed-runner:0.1.0` and `ghcr.io/ryvion/transcode-runner:0.1.0` because `hub-orch` currently references those tags directly.
- It also publishes `latest` and the commit SHA tags for operator testing.
