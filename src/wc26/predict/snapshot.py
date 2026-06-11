"""Append-only prediction snapshots (plan section 7).

Every emitted prediction is recorded with epoch, as-of timestamp and model
version so pre-lineup / post-lineup / final comparisons are queries, not
estimates.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from wc26.paths import configs_dir, snapshots_dir


def model_version() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        sha = "nogit"
    cfg = (configs_dir() / "model.yaml").read_bytes()
    return f"{sha}+{hashlib.sha256(cfg).hexdigest()[:8]}"


def write_snapshot(prediction: dict, epoch: str = "pre_lineup", path: Path | None = None) -> None:
    rec = {k: v for k, v in prediction.items() if k != "matrix"}
    rec.update(
        epoch=epoch,
        snapshot_ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model_version=model_version(),
    )
    snapshots_dir().mkdir(parents=True, exist_ok=True)
    target = path or snapshots_dir() / "predictions.jsonl"
    with open(target, "a") as f:
        f.write(json.dumps(rec) + "\n")
