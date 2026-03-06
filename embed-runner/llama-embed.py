import argparse
import json
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = os.environ.get("RYV_EMBED_MODEL", "/opt/models/all-MiniLM-L6-v2")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a text embedding and write it to a JSON file.")
    parser.add_argument("--input", default=os.environ.get("RYV_EMBED_INPUT", "/work/input"))
    parser.add_argument("--output", default=os.environ.get("RYV_EMBED_OUTPUT", "/work/output"))
    return parser.parse_args()


def main():
    args = parse_args()
    text = Path(args.input).read_text(encoding="utf-8")
    model = SentenceTransformer(DEFAULT_MODEL)
    vector = model.encode(text, normalize_embeddings=True).tolist()
    payload = {
        "model": DEFAULT_MODEL,
        "dimensions": len(vector),
        "embedding": vector,
    }
    Path(args.output).write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
