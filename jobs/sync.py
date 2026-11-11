"""Move files between a Hugging Face Job and the private runs repo ($RUNS_REPO), at the same paths on both sides.

    python jobs/sync.py pull "data/processed/**" "outputs/ckpt/A/**"   # download matching files
    python jobs/sync.py push data/processed outputs/results             # upload files or folders
    python jobs/sync.py wait data/processed/harmful/hinglish/dev.jsonl --minutes 120   # one deadline for all paths

push refuses the folders that hold raw model generations or human-label sheets (they contain harmful
completions and must never be uploaded) and merged repaired checkpoints, and under outputs/factors it
allows only each pass's meta.json, never the factor tensors. It skips paths that don't exist.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from huggingface_hub import file_exists, snapshot_download, upload_file, upload_folder

NEVER_PUSH = ("outputs/generations", "outputs/labels", ".cache", "outputs/ckpt/repaired")
FACTORS = "outputs/factors"  # tens of GB of tensors: only each pass's meta.json may leave the job
POLL_SECONDS = 60


def runs_repo() -> str:
    """The private runs repo this job reads from and writes to, set by jobs/launch.py."""
    return os.environ["RUNS_REPO"]


def pull(patterns: list[str]) -> None:
    """Download the runs-repo files matching `patterns` into the current folder."""
    snapshot_download(runs_repo(), repo_type="dataset", allow_patterns=patterns, local_dir=".")


def _is_under(path: str, root: str) -> bool:
    """True if `path` is `root` or inside it."""
    return path == root or path.startswith(root + "/")


def is_forbidden(path: str) -> bool:
    """True for a path that must never be uploaded: see the module docstring."""
    normalised = Path(os.path.normpath(path)).as_posix()
    if _is_under(normalised, FACTORS):
        return not normalised.endswith("/meta.json")
    return any(_is_under(normalised, root) for root in NEVER_PUSH)


def push(paths: list[str]) -> None:
    """Upload each existing file or folder to the same path in the runs repo."""
    message = f"{os.environ.get('STAGE', 'manual')} {os.environ.get('STAGE_ARG', '')}".strip()
    for path in paths:
        if is_forbidden(path):
            raise SystemExit(f"refusing to push {path}: generations, label sheets, checkpoints or factor tensors")
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


def wait(paths: list[str], minutes: float) -> bool:
    """Poll the runs repo until every path exists there. Returns False if `minutes` pass first: one deadline
    for all of them, so a missing file can't stretch the wait to len(paths) × minutes."""
    deadline = time.time() + 60 * minutes
    missing = list(paths)
    while True:
        missing = [path for path in missing if not file_exists(runs_repo(), path, repo_type="dataset")]
        if not missing:
            return True
        if time.time() > deadline:
            print(f"wait: still missing after {minutes:g} min: {', '.join(missing)}")
            return False
        time.sleep(POLL_SECONDS)


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
        found = wait(args.paths, args.minutes)
        return 0 if found else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
