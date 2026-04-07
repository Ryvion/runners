"""Ryvion embedding runner.

Handles both single text and batch embedding.

Single mode: reads text from --input file, writes embedding to --output file.
Batch mode: reads /work/job.json with task="batch_embed" and chunks array,
            writes all embeddings to /work/output as JSON.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = os.environ.get("RYV_EMBED_MODEL", "/opt/models/all-MiniLM-L6-v2")


def run_single(input_path, output_path):
    """Embed a single text file."""
    text = Path(input_path).read_text(encoding="utf-8")
    model = SentenceTransformer(DEFAULT_MODEL)
    vector = model.encode(text, normalize_embeddings=True).tolist()
    payload = {
        "model": DEFAULT_MODEL,
        "dimensions": len(vector),
        "embedding": vector,
    }
    Path(output_path).write_text(json.dumps(payload), encoding="utf-8")


def run_batch():
    """Embed multiple text chunks from job.json. Writes embeddings to /work/output."""
    job_path = Path("/work/job.json")
    if not job_path.exists():
        print("ERROR: no /work/job.json for batch mode", file=sys.stderr)
        return False

    job = json.loads(job_path.read_text(encoding="utf-8"))
    chunks = job.get("chunks", [])
    if not chunks:
        print("ERROR: no chunks in job spec", file=sys.stderr)
        return False

    print(json.dumps({"event": "batch_embed_start", "chunks": len(chunks)}), file=sys.stderr)

    model = SentenceTransformer(DEFAULT_MODEL)
    vectors = model.encode(chunks, normalize_embeddings=True, show_progress_bar=True)

    embeddings = [v.tolist() for v in vectors]

    output = {
        "model": DEFAULT_MODEL,
        "dimensions": len(embeddings[0]) if embeddings else 0,
        "count": len(embeddings),
        "embeddings": embeddings,
        "kb_id": job.get("kb_id", ""),
        "doc_id": job.get("doc_id", ""),
    }
    Path("/work/output").write_text(json.dumps(output), encoding="utf-8")

    print(json.dumps({
        "event": "batch_embed_done",
        "chunks": len(chunks),
        "dimensions": output["dimensions"],
    }), file=sys.stderr)
    return True


def main():
    # Check if this is a batch embed job
    job_path = Path("/work/job.json")
    if job_path.exists():
        try:
            job = json.loads(job_path.read_text(encoding="utf-8"))
            if job.get("task") == "batch_embed":
                success = run_batch()
                sys.exit(0 if success else 1)
        except (json.JSONDecodeError, KeyError):
            pass  # Not a batch job, fall through to single mode

    # Single text mode (original behavior)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.environ.get("RYV_EMBED_INPUT", "/work/input"))
    parser.add_argument("--output", default=os.environ.get("RYV_EMBED_OUTPUT", "/work/output"))
    args = parser.parse_args()
    run_single(args.input, args.output)


if __name__ == "__main__":
    main()
