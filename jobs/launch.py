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
    .venv/bin/python jobs/launch.py calibrate --gpu a10g-small   # when no L4 is free; the chain inherits it
    .venv/bin/python jobs/launch.py pencil A            # factors, eigendecomposition, repair, baselines (by hand)

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
DEFAULT_GPU = "l4x1"
GPU_CHOICES = ("l4x1", "a10g-small")  # both 24 GB; anything bigger is a deliberate code change
GPU_DISK_GB = {"l4x1": 400, "a10g-small": 110}  # job disk per machine, from `hf jobs hardware`
CPU_FLAVOR = "cpu-basic"

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
    gpu: bool
    timeout: str
    needs_arg: bool = False
    then: tuple[str, ...] = ()  # "stage" or "stage:arg", started by the job when this stage succeeds
    min_disk_gb: int = 0  # refuse machines with less job disk than this


STAGES = {
    "calibrate": Stage(gpu=True, timeout="90m", then=("hinglish", "finetune:A", "finetune:C", "finetune:D")),
    "hinglish": Stage(gpu=False, timeout="2h", then=("evaluate:theta0",)),
    "finetune": Stage(gpu=True, timeout="5h", needs_arg=True),
    "evaluate": Stage(gpu=True, timeout="3h", needs_arg=True),
    # ~220 GB of fp32 factors, ~50 GB of model downloads and up to six 7 GB repaired checkpoints
    "pencil": Stage(gpu=True, timeout="9h", needs_arg=True, min_disk_gb=300),
}


def flavor_for(stage: Stage, gpu: str) -> str:
    """The machine a stage runs on: the chosen GPU for GPU stages, the basic CPU machine otherwise."""
    if stage.gpu:
        return gpu
    return CPU_FLAVOR


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


def start_job(stage_name: str, arg: str, code_repo: str, runs_repo: str, version: str, gpu: str = DEFAULT_GPU):
    """Start one stage's job and return its JobInfo. GPU stages run on `gpu`, the others on the CPU machine."""
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
        "GPU_FLAVOR": gpu,  # so the stages this job starts run on the same GPU
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
        flavor=flavor_for(stage, gpu),
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
    parser.add_argument(
        "--gpu",
        choices=GPU_CHOICES,
        default=None,
        help=f"GPU for this stage and the ones it starts (default {DEFAULT_GPU}; a chained launch keeps its parent's)",
    )
    args = parser.parse_args(argv)
    if args.stage != "status" and STAGES[args.stage].needs_arg and not args.arg:
        parser.error(f"{args.stage} needs an argument, e.g. `{args.stage} A`")
    return args


def chosen_gpu(args: argparse.Namespace) -> str:
    """--gpu if given; in a chained launch, the GPU the starting job ran with; otherwise DEFAULT_GPU."""
    if args.gpu:
        return args.gpu
    if args.chained:
        return os.environ.get("GPU_FLAVOR", DEFAULT_GPU)
    return DEFAULT_GPU


def check_disk(stage_name: str, gpu: str) -> None:
    """Stop before launching a GPU stage on a machine whose disk can't hold its files."""
    stage = STAGES[stage_name]
    if not stage.gpu:
        return
    available = GPU_DISK_GB[gpu]
    if stage.min_disk_gb > available:
        raise SystemExit(f"{stage_name} needs about {stage.min_disk_gb} GB of disk; {gpu} has {available} GB")


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.stage == "status":
        show_status()
        return
    gpu = chosen_gpu(args)
    check_disk(args.stage, gpu)
    code_repo, runs_repo = repo_names()
    if args.chained:
        version = os.environ.get("PENCIL_GIT_HASH", "unknown")
    else:
        warn_about_uncommitted_code()
        upload_code(code_repo, runs_repo)
        version = code_version()
    job = start_job(args.stage, args.arg, code_repo, runs_repo, version, gpu)
    print(f"started {args.stage} {args.arg} on {flavor_for(STAGES[args.stage], gpu)}: {job.url}")
    print(f"  log: .venv/bin/hf jobs logs {job.id}")


if __name__ == "__main__":
    main()
