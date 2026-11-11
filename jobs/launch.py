#!/usr/bin/env python
"""Start a pipeline stage as a Hugging Face Job. Run it on your machine, from the repo root.

A launch uploads the code to the private dataset repo <you>/safety-pencil-code (only changed files travel) and
starts the stage's job. The job downloads that code, installs the pinned environment (jobs/entrypoint.sh), pulls
its inputs from the private dataset repo <you>/safety-pencil-runs, runs jobs/stages/<stage>.sh and pushes its
outputs back there. Raw model generations never leave the job. When a stage succeeds, its job starts the stages
listed in `then` below, so `calibrate` alone sets off hinglish and the three fine-tunes.

    .venv/bin/python jobs/launch.py calibrate
    .venv/bin/python jobs/launch.py finetune A          # A, C or D; also probe-evaluates its four adapters
    .venv/bin/python jobs/launch.py evaluate theta0     # theta0, or A / C / D for a run's four adapters
    .venv/bin/python jobs/launch.py status              # recent jobs and their state

Follow a job's log with `.venv/bin/hf jobs logs <job id>`. Log in once first with `.venv/bin/hf auth login`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_REPO_NAME = "safety-pencil-code"
RUNS_REPO_NAME = "safety-pencil-runs"
IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm"
HUB_CLIENT = "huggingface_hub==2.4.0"
PROJECT_LABEL = "safety-pencil"

# What a launch uploads as "the code": everything a job runs or tests, nothing it produces.
CODE_FILES = [
    "src/**",
    "scripts/**",
    "configs/**",
    "jobs/**",
    "tests/**",
    "patches/**",
    "pyproject.toml",
    "requirements-*",
    "README.md",
]
NOT_CODE = ["**/__pycache__/**", "**/*.pyc", "**/*.egg-info/**", ".env"]

# Inside the job: fetch the code, then hand over to jobs/entrypoint.sh.
BOOTSTRAP = (
    'uvx --from "$HUB_CLIENT" hf download "$CODE_REPO" --repo-type dataset --local-dir /work/code'
    " && exec bash /work/code/jobs/entrypoint.sh"
)


@dataclass(frozen=True)
class Stage:
    flavor: str
    timeout: str
    needs_arg: bool = False
    then: tuple[str, ...] = ()  # "stage" or "stage:arg", started by the job when this stage succeeds


STAGES = {
    "calibrate": Stage("l4x1", "90m", then=("hinglish", "finetune:A", "finetune:C", "finetune:D")),
    "hinglish": Stage("cpu-basic", "2h", then=("evaluate:theta0",)),
    "finetune": Stage("l4x1", "5h", needs_arg=True),
    "evaluate": Stage("l4x1", "3h", needs_arg=True),
}


def repo_names() -> tuple[str, str]:
    """(code repo, runs repo) under the logged-in account."""
    from huggingface_hub import whoami

    namespace = whoami()["name"]
    return f"{namespace}/{CODE_REPO_NAME}", f"{namespace}/{RUNS_REPO_NAME}"


def code_version() -> str:
    """The commit the job's results will record (pencil.results.git_hash on this machine)."""
    # imported here, not at the top: chained launches inside a job run without pencil installed
    from pencil.results import git_hash

    return git_hash()


def warn_about_uncommitted_code() -> None:
    """Results record the commit, so say when the uploaded code differs from it."""
    command = ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", "src", "scripts", "configs", "jobs"]
    changes = subprocess.run(command, capture_output=True, text=True).stdout.strip()
    if changes:
        print("note: uploading code that differs from the last commit (results will say so):")
        print(changes)


def upload_code(code_repo: str, runs_repo: str) -> None:
    """Mirror the code into the private code repo and make sure the private runs repo exists."""
    from huggingface_hub import create_repo, upload_folder

    create_repo(code_repo, repo_type="dataset", private=True, exist_ok=True)
    create_repo(runs_repo, repo_type="dataset", private=True, exist_ok=True)
    upload_folder(
        repo_id=code_repo,
        repo_type="dataset",
        folder_path=REPO_ROOT,
        allow_patterns=CODE_FILES,
        ignore_patterns=NOT_CODE,
        delete_patterns=CODE_FILES,  # files deleted here disappear from the code repo too
        commit_message=f"code at {code_version()}",
    )


def start_job(stage_name: str, arg: str, code_repo: str, runs_repo: str, version: str):
    """Start one stage's job and return its JobInfo."""
    from huggingface_hub import get_token, run_job

    stage = STAGES[stage_name]
    env = {
        "HUB_CLIENT": HUB_CLIENT,
        "CODE_REPO": code_repo,
        "RUNS_REPO": runs_repo,
        "STAGE": stage_name,
        "STAGE_ARG": arg,
        "THEN": " ".join(stage.then),
        "PENCIL_GIT_HASH": version,
    }
    labels = {"project": PROJECT_LABEL, "stage": stage_name}
    name = f"pencil-{stage_name}"
    if arg:
        labels["arg"] = arg
        name = f"{name}-{arg}"
    return run_job(
        image=IMAGE,
        command=["bash", "-c", BOOTSTRAP],
        env=env,
        secrets={"HF_TOKEN": get_token()},
        flavor=stage.flavor,
        timeout=stage.timeout,
        name=name,
        labels=labels,
    )


def show_status(limit: int = 15) -> None:
    """Print this project's most recent jobs, newest first."""
    from huggingface_hub import list_jobs

    jobs = list_jobs(labels={"project": PROJECT_LABEL})
    jobs = sorted(jobs, key=lambda job: job.created_at.timestamp(), reverse=True)[:limit]
    for job in jobs:
        labels = job.labels or {}
        what = f"{labels.get('stage', '?')} {labels.get('arg', '')}".strip()
        print(f"{job.id}  {job.created_at:%d %b %H:%M}  {job.status.stage:<10}  {what}")


def parse_args(argv=None) -> argparse.Namespace:
    """The stage to start (or `status`) and its argument; finetune and evaluate need one."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=[*STAGES, "status"])
    parser.add_argument("arg", nargs="?", default="", help="the run (A, C, D) or theta0, for finetune / evaluate")
    parser.add_argument(
        "--chained", action="store_true", help="started by a finished job: reuse the uploaded code and its version"
    )
    args = parser.parse_args(argv)
    if args.stage != "status" and STAGES[args.stage].needs_arg and not args.arg:
        parser.error(f"{args.stage} needs an argument, e.g. `{args.stage} A`")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.stage == "status":
        show_status()
        return
    code_repo, runs_repo = repo_names()
    if args.chained:
        version = os.environ.get("PENCIL_GIT_HASH", "unknown")
    else:
        warn_about_uncommitted_code()
        upload_code(code_repo, runs_repo)
        version = code_version()
    job = start_job(args.stage, args.arg, code_repo, runs_repo, version)
    print(f"started {args.stage} {args.arg} on {STAGES[args.stage].flavor}: {job.url}")
    print(f"  log: .venv/bin/hf jobs logs {job.id}")


if __name__ == "__main__":
    main()
