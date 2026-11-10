"""Result files: one JSON per result (CLAUDE.md rule 6).

Each file wraps the result payload with the provenance needed to reproduce it: the config, the seed, the
git commit (marked "-dirty" when tracked files had uncommitted changes), the checkpoint id and a timestamp.
Tensors and numpy values in the payload are converted to plain lists and numbers.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, outputs_dir


def git_hash() -> str:
    """The repo's short commit hash, with "-dirty" appended when tracked files have uncommitted changes.

    Returns "unknown" when REPO_ROOT is not inside a git checkout or git is not installed. Untracked files
    do not count as changes."""
    commit = _git_output("rev-parse", "--short", "HEAD")
    if commit is None:
        return "unknown"
    changes = _git_output("status", "--porcelain", "--untracked-files=no")
    if changes:
        return f"{commit}-dirty"
    return commit


def _git_output(*git_args: str) -> str | None:
    """Stripped output of `git -C REPO_ROOT <git_args>`, or None if git is missing or the command fails."""
    command = ["git", "-C", str(REPO_ROOT), *git_args]
    try:
        output = subprocess.check_output(command, stderr=subprocess.DEVNULL, text=True, errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None
    return output.strip()


def write_result(
    name: str,
    payload: dict[str, Any],
    *,
    config: dict | None = None,
    seed: int | None = None,
    checkpoint: str | None = None,
    out_dir: str | Path | None = None,
) -> Path:
    """Write {name}.json (payload plus provenance) to `out_dir`, default outputs/results/. Returns its path."""
    record = {
        "name": name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": git_hash(),
        "seed": seed,
        "checkpoint": checkpoint,
        "config": config,
        "result": payload,
    }
    folder = Path(out_dir) if out_dir else outputs_dir("results")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.json"
    path.write_text(json.dumps(record, indent=1, default=_jsonable))
    return path


def _jsonable(value: Any):
    """JSON fallback: tensors and numpy arrays become lists, numpy scalars numbers, anything else its str()."""
    try:
        import numpy as np
        import torch
    except ImportError:
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
