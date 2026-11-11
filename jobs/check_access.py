"""Check that this Hugging Face token can read every model in configs/experiment.yaml (all are gated).

Fails if the aligned model θ0 is unreadable, since nothing runs without it. Any other unreadable model is
reported loudly but does not fail: the fine-tunes don't need the judges or the base model, so they can start
while the licence is accepted on the Hub. Writes outputs/results/calibration_access.json.
"""

from __future__ import annotations

import sys

from huggingface_hub import hf_hub_download

from pencil.config import load_config
from pencil.results import write_result

REQUIRED_ROLE = "aligned"


def readable(repo_id: str) -> str:
    """'ok', or the error's class name if the model's config.json can't be downloaded."""
    try:
        hf_hub_download(repo_id, "config.json")
    except Exception as error:  # GatedRepoError, RepositoryNotFoundError, network errors: all mean "not usable"
        return type(error).__name__
    return "ok"


def main() -> int:
    models = load_config()["models"]
    access = {}
    for role, repo_id in models.items():
        access[role] = {"repo": repo_id, "status": readable(repo_id)}
        print(f"{role:<14} {repo_id:<40} {access[role]['status']}")
    write_result("calibration_access", access)

    unreadable = [role for role, entry in access.items() if entry["status"] != "ok"]
    for role in unreadable:
        print(f"WARNING: cannot read {models[role]}: accept its licence on the Hub with this account")
    if REQUIRED_ROLE in unreadable:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
