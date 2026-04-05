"""Ryvion Whisper transcription runner.

Reads /work/job.json, transcribes the audio file with OpenAI Whisper,
and writes /work/output.json, /work/receipt.json, and /work/metrics.json.
"""

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

OUTPUT_PATH = Path("/work/output.json")
RECEIPT_PATH = Path("/work/receipt.json")
METRICS_PATH = Path("/work/metrics.json")

MODELS = {
    "whisper-1": "base",
    "whisper-large": "large-v3",
}

DEFAULT_MODEL = "whisper-1"


def load_job(path: Path = Path("/work/job.json")) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    job = load_job()

    model_name = str(job.get("model") or DEFAULT_MODEL).strip()
    whisper_model = MODELS.get(model_name, MODELS[DEFAULT_MODEL])
    language = str(job.get("language") or "").strip() or None

    # Find audio file — either mounted at /work/audio or via audio_key
    audio_path = None
    audio_key = str(job.get("audio_key") or "").strip()
    if audio_key:
        # The node mounts blobs at /work/blobs/
        candidate = Path("/work/blobs") / audio_key
        if candidate.exists():
            audio_path = str(candidate)

    if not audio_path:
        # Check prefetched audio (downloaded by node-agent before container start)
        prefetched = Path("/work/input_audio")
        if prefetched.exists():
            audio_path = str(prefetched)

    if not audio_path:
        # Try common locations
        for ext in ["wav", "mp3", "m4a", "flac", "ogg", "webm", "mp4"]:
            candidate = Path(f"/work/input.{ext}")
            if candidate.exists():
                audio_path = str(candidate)
                break

    if not audio_path:
        # Check /work/ for any audio file
        for f in Path("/work").iterdir():
            if f.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4", ""}:
                if f.name in ("job.json", "receipt.json", "metrics.json"):
                    continue
                audio_path = str(f)
                break

    if not audio_path:
        print("ERROR: no audio file found", file=sys.stderr)
        return 1

    print(f"Transcribing {audio_path} with model={whisper_model}")
    started = time.time()

    import whisper

    model = whisper.load_model(whisper_model)
    result = model.transcribe(audio_path, language=language)
    text = result.get("text", "").strip()
    duration_ms = int((time.time() - started) * 1000)

    output = {
        "text": text,
        "language": result.get("language", ""),
        "segments": [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            }
            for seg in result.get("segments", [])
        ],
    }

    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    output_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    RECEIPT_PATH.write_text(json.dumps({
        "output_hash": output_hash,
        "ok": True,
        "model": model_name,
        "whisper_model": whisper_model,
        "language": result.get("language", ""),
        "duration_ms": duration_ms,
        "text": text,
    }), encoding="utf-8")

    METRICS_PATH.write_text(json.dumps({
        "engine": "whisper",
        "model": whisper_model,
        "duration_ms": duration_ms,
        "audio_file": os.path.basename(audio_path),
        "text_length": len(text),
    }), encoding="utf-8")

    print(json.dumps({
        "output_hash": output_hash,
        "ok": True,
        "model": model_name,
        "text_length": len(text),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
