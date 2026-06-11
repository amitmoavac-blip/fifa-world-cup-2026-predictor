from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("WC26_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("could not locate repo root; set WC26_ROOT")


def configs_dir() -> Path:
    return repo_root() / "configs"


def raw_dir() -> Path:
    return repo_root() / "data" / "raw"


def processed_dir() -> Path:
    return repo_root() / "data" / "processed"


def snapshots_dir() -> Path:
    return repo_root() / "data" / "snapshots"


def reports_dir() -> Path:
    return repo_root() / "reports"
