"""Move files between a Hugging Face Job and the private runs repo ($RUNS_REPO), at the same paths on both sides.

    python jobs/sync.py pull "data/processed/**" "outputs/ckpt/A/**"   # download matching files
    python jobs/sync.py push data/processed outputs/results             # upload files or folders
    python jobs/sync.py wait data/processed/harmful/hinglish/dev.jsonl --minutes 120

push refuses the folders that hold raw model generations or human-label sheets (they contain harmful
completions and must never be uploaded); it skips paths that don't exist.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from huggingface_hub import file_exists, snapshot_download, upload_file, upload_folder

NEVER_PUSH = ("outputs/generations", "outputs/labels", ".cache")
POLL_SECONDS = 60


def runs_repo() -> str:
    """The private runs repo this job reads from and writes to, set by jobs/launch.py."""
    return os.environ["RUNS_REPO"]


def pull(patterns: list[str]) -> None:
    """Download the runs-repo files matching `patterns` into the current folder."""
    snapshot_download(runs_repo(), repo_type="dataset", allow_patterns=patterns, local_dir=".")


def is_forbidden(path: str) -> bool:
    normalised = Path(os.path.normpath(path)).as_posix()
    return any(normalised == root or normalised.startswith(root + "/") for root in NEVER_PUSH)


def push(paths: list[str]) -> None:
    """Upload each existing file or folder to the same path in the runs repo."""
    message = f"{os.environ.get('STAGE', 'manual')} {os.environ.get('STAGE_ARG', '')}".strip()
    for path in paths:
        if is_forbidden(path):
            raise SystemExit(f"refusing to push {path}: it can hold raw generations or label sheets")
        local = Path(path)
        if not local.exists():
            print(f"push: {path} does not exist, skipped")
            continue
        if local.is_dir():
            upload_folder(
                repo_id=runs_repo(),
                repo_type="dataset",
                folder_path=local,
                path_in_repo=local.as_posix(),
                commit_message=f"{message}: {path}",
            )
        else:
            upload_file(
                repo_id=runs_repo(),
                repo_type="dataset",
                path_or_fileobj=local,
                path_in_repo=local.as_posix(),
                commit_message=f"{message}: {path}",
            )
        print(f"pushed {path}")


def wait(path: str, minutes: float) -> bool:
    """Poll the runs repo until `path` exists there. Returns False if `minutes` pass first."""
    deadline = time.time() + 60 * minutes
    while not file_exists(runs_repo(), path, repo_type="dataset"):
        if time.time() > deadline:
            print(f"wait: {path} did not appear within {minutes:g} min")
            return False
        time.sleep(POLL_SECONDS)
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["pull", "push", "wait"])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--minutes", type=float, default=120, help="how long `wait` polls")
    args = parser.parse_args(argv)
    if args.action == "pull":
        pull(args.paths)
    elif args.action == "push":
        push(args.paths)
    else:
        found = all(wait(path, args.minutes) for path in args.paths)
        return 0 if found else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
