"""The run ledger: one record per Hugging Face Job, kept in the runs repo next to the results the job wrote.

    python jobs/ledger.py start                   # at the start of a job (jobs/entrypoint.sh)
    python jobs/ledger.py finish <exit status>    # at its end, whether the stage succeeded or not

Both read the job's run id from $PENCIL_RUN_ID (e.g. "20261112T001500Z-pencil-A"); results.write_result stamps
the same id into every result, which is how `finish` finds the results this job wrote. `finish` uploads
runs/<run id>/ with run.json (stage, argument, code commit, machine, start and end, outcome, results), log.txt
(the job's output, progress bars squeezed) and results/, an unchangeable copy of each result: a later rerun
that overwrites outputs/results/<name>.json loses nothing. jobs/report.py turns the ledger into a summary.
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

RUN_ID_ENV = "PENCIL_RUN_ID"
LEDGER_DIR = Path("outputs/runs")
RESULTS_DIR = Path("outputs/results")
LOG_NAME = "log.txt"
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"  # UTC
MAX_LOG_LINES = 20000  # the end of the log is kept; the HF job page has the rest while it lasts
TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{20,}")  # never let a token into the log


def run_folder(run_id: str) -> Path:
    """outputs/runs/<run id>/, where the job's record, log and result copies are gathered."""
    return LEDGER_DIR / run_id


def new_record(run_id: str, environment: dict, now: float) -> dict:
    """The record a job starts with: what runs, on which code and machine, and when."""
    return {
        "run_id": run_id,
        "stage": environment.get("STAGE", ""),
        "arg": environment.get("STAGE_ARG", ""),
        "code": environment.get("PENCIL_GIT_HASH", "unknown"),
        "machine": environment.get("GPU_FLAVOR", ""),
        "gpu": environment.get("GPU_NAME", ""),
        "started": _utc(now),
        "status": "running",
    }


def results_of(run_id: str, results_dir: Path = RESULTS_DIR) -> list[Path]:
    """The result files this run wrote: those whose run_id field is this run's (pulled results are others')."""
    paths = []
    for path in sorted(results_dir.glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("run_id") == run_id:
            paths.append(path)
    return paths


def finish_record(record: dict, exit_status: int, results: list[Path], now: float) -> dict:
    """The record with the job's outcome, duration and result names added."""
    started = calendar.timegm(time.strptime(record["started"], TIME_FORMAT))
    status = "succeeded"
    if exit_status != 0:
        status = f"failed (exit {exit_status})"
    finished = dict(record)
    finished.update(
        finished=_utc(now),
        minutes=round((now - started) / 60, 1),
        status=status,
        results=[path.stem for path in results],
    )
    return finished


def clean_log(text: str) -> str:
    """The log with each progress bar reduced to its last state, tokens masked, and only its last lines."""
    # split on "\n" only: str.splitlines() would also split at the "\r" a progress bar redraws itself with
    lines = [line.split("\r")[-1] for line in text.rstrip("\n").split("\n")]
    kept = lines[-MAX_LOG_LINES:]
    return TOKEN_PATTERN.sub("hf_***", "\n".join(kept)) + "\n"


def start(run_id: str) -> None:
    """Write the starting record and upload it, so even a job that dies early shows up in the ledger."""
    folder = run_folder(run_id)
    folder.mkdir(parents=True, exist_ok=True)
    record = new_record(run_id, dict(os.environ), time.time())
    (folder / "run.json").write_text(json.dumps(record, indent=1))
    _upload(folder, run_id, f"start {run_id}")


def finish(run_id: str, exit_status: int) -> None:
    """Complete the record, gather the log and copies of this job's results, and upload them together."""
    folder = run_folder(run_id)
    record = json.loads((folder / "run.json").read_text())
    results = results_of(run_id)
    copies = folder / "results"
    copies.mkdir(exist_ok=True)
    for path in results:
        shutil.copy2(path, copies / path.name)
    log = folder / LOG_NAME
    if log.exists():
        log.write_text(clean_log(log.read_text(errors="replace")))
    record = finish_record(record, exit_status, results, time.time())
    (folder / "run.json").write_text(json.dumps(record, indent=1))
    _upload(folder, run_id, f"{record['status']}: {run_id}")


def _upload(folder: Path, run_id: str, message: str) -> None:
    """Upload outputs/runs/<run id>/ to runs/<run id>/ in the runs repo."""
    from huggingface_hub import upload_folder

    repo = os.environ["RUNS_REPO"]
    upload_folder(repo_id=repo, repo_type="dataset", folder_path=folder, path_in_repo=f"runs/{run_id}")
    print(f"ledger: {message}")


def _utc(seconds: float) -> str:
    """A UTC timestamp like 2026-11-12T00:15:00Z."""
    return time.strftime(TIME_FORMAT, time.gmtime(seconds))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["start", "finish"])
    parser.add_argument("exit_status", nargs="?", type=int, default=0, help="finish: the stage's exit status")
    args = parser.parse_args(argv)
    run_id = os.environ[RUN_ID_ENV]
    if args.action == "start":
        start(run_id)
    else:
        finish(run_id, args.exit_status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
