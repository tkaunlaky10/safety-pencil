"""Configuration and environment loading.

REPO_ROOT is the checkout this package lives in; configs/experiment.yaml is the default experiment config;
secrets such as HF_TOKEN come from .env at the repo root.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment.yaml"


def load_env() -> None:
    """Load .env (HF_TOKEN etc.) and turn off AuditKIT's raw-output run cache by default."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional at import time
        pass
    else:
        load_dotenv(REPO_ROOT / ".env")
    os.environ.setdefault("AUDITKIT_DISABLE_RUN_CACHE", "1")


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """The experiment config as a dict, read from `path` (default configs/experiment.yaml)."""
    if not path:
        path = DEFAULT_CONFIG
    with open(path) as file:
        return yaml.safe_load(file)


def outputs_dir(*parts: str) -> Path:
    """The folder outputs/parts... under the repo root, created if it does not exist."""
    path = REPO_ROOT.joinpath("outputs", *parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
