"""One JSON per result, with config, seed, git hash, checkpoint id and timestamp (CLAUDE.md rule 6)."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, outputs_dir


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def write_result(
    name: str,
    payload: dict[str, Any],
    *,
    config: dict | None = None,
    seed: int | None = None,
    checkpoint: str | None = None,
    out_dir: str | Path | None = None,
) -> Path:
    record = {
        "name": name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": git_hash(),
        "seed": seed,
        "checkpoint": checkpoint,
        "config": config,
        "result": payload,
    }
    d = Path(out_dir) if out_dir else outputs_dir("results")
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.json"
    path.write_text(json.dumps(record, indent=1, default=_jsonable))
    return path


def _jsonable(x: Any):
    try:
        import numpy as np
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
    except ImportError:
        pass
    return str(x)
