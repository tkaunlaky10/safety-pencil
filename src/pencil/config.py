"""Configuration and environment loading."""
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
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:  # python-dotenv is optional at import time
        pass
    os.environ.setdefault("AUDITKIT_DISABLE_RUN_CACHE", "1")


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def outputs_dir(*parts: str) -> Path:
    p = REPO_ROOT.joinpath("outputs", *parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
