"""Ryvion starter agent runner.

This image is intentionally small: it proves the persistent agent-hosting
contract, exposes the deployment environment, and stays alive until the hub
asks node-agent to stop the container.
"""

import hashlib
import json
import os
import signal
import sys
import time

shutdown = False


def handle_shutdown(signum, _frame):
    global shutdown
    shutdown = True
    print(json.dumps({"event": "shutdown_signal", "signal": signum}), flush=True)


signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)


def load_job(path="/work/job.json"):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(json.dumps({"event": "job_load_failed", "error": str(exc)}), file=sys.stderr, flush=True)
        return {}


def write_receipt(started_at, job):
    uptime_seconds = int(time.time() - started_at)
    receipt = {
        "output_hash": hashlib.sha256(f"{os.environ.get('RYVION_DEPLOYMENT_ID', '')}:{uptime_seconds}".encode()).hexdigest(),
        "deployment_id": os.environ.get("RYVION_DEPLOYMENT_ID", ""),
        "model": os.environ.get("RYVION_MODEL", job.get("model", "")),
        "kb_ids": os.environ.get("RYVION_KB_IDS", job.get("kb_ids", "")),
        "uptime_seconds": uptime_seconds,
        "status": "stopped" if shutdown else "exited",
    }
    with open("/work/receipt.json", "w", encoding="utf-8") as f:
        json.dump(receipt, f)


def main():
    started_at = time.time()
    job = load_job()
    heartbeat_seconds = int(os.environ.get("RYVION_AGENT_LOG_INTERVAL_SECONDS", "30"))
    if heartbeat_seconds < 5:
        heartbeat_seconds = 5

    print(json.dumps({
        "event": "agent_started",
        "deployment_id": os.environ.get("RYVION_DEPLOYMENT_ID", ""),
        "hub_url": os.environ.get("RYVION_HUB_URL", ""),
        "mcp_url": os.environ.get("RYVION_MCP_URL", ""),
        "kb_ids": os.environ.get("RYVION_KB_IDS", job.get("kb_ids", "")),
        "model": os.environ.get("RYVION_MODEL", job.get("model", "")),
    }), flush=True)

    while not shutdown:
        time.sleep(heartbeat_seconds)
        print(json.dumps({
            "event": "agent_heartbeat",
            "deployment_id": os.environ.get("RYVION_DEPLOYMENT_ID", ""),
            "uptime_seconds": int(time.time() - started_at),
        }), flush=True)

    write_receipt(started_at, job)
    print(json.dumps({"event": "agent_stopped"}), flush=True)


if __name__ == "__main__":
    main()
