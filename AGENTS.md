# Codex Agent Instructions — runners

## Project Context

Container images for GPU/CPU workloads. Built by GitHub Actions, pushed to `ghcr.io/ryvion/*`.

## Container Contract

All containers:
1. Read `/work/job.json`
2. Write `/work/receipt.json` (output hash + metadata)
3. Write `/work/metrics.json` (must include `output_name`)
4. Write output artifact to `/work/`

## Network Policy

- ALL containers: `--network=none` (no internet)
- EXCEPT finetune-runner: `--network=bridge` (HuggingFace downloads)
- EXCEPT agent containers: `--network=bridge` (hub API access)

## Key Rules

- Never install packages that override pinned torch version
- GPU imports fail in CI — verify with `pip list`, not Python imports
- All runners use SIGTERM handlers (exit 143)
- `metrics.json` must have `output_name` field
- Commits: Keep messages SHORT, no Co-Authored-By
