"""Make the public Hugging Face repos for the submission (SUBMISSION_TODO.md steps 5–6).

    python jobs/publish.py --dry-run     # print the plan: every repo, its files and sizes; writes nothing
    python jobs/publish.py               # create the repos as public and upload

Sources: the private runs repo (tkaunlaky10/safety-pencil-runs, which stays private) and a few files that exist
only in this checkout (CPU-only results and summary.md). Targets:

    tkaunlaky10/safety-pencil-data       dataset  data/processed, minus the BeaverTails rows and CuratorKIT rejects
    tkaunlaky10/safety-pencil-results    dataset  outputs/results, outputs/figures, runs/, summary.md
    tkaunlaky10/safety-pencil-lora-run{A,C,D}  model  the adapters at frac25 … frac100

Every file passes `is_forbidden` before it is planned, and again before upload.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from publish_cards import DATA_CARD, RESULTS_CARD, model_card  # noqa: E402

OWNER = "tkaunlaky10"
SOURCE_REPO = f"{OWNER}/safety-pencil-runs"
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = ("A", "C", "D")
FRACTIONS = (25, 50, 75, 100)
LOCAL_ONLY = (  # computed on CPU after the jobs, never uploaded to the runs repo
    "outputs/results/p6_frontier_runA.json",
    "outputs/results/wiseft_shape_runA.json",
    "outputs/figures/wiseft_shape_runA.png",
)
# never public: harmful completions (BeaverTails), generations, label sheets, translation review sheets,
# CuratorKIT rejects, checkpoints we did not train as adapters, factor tensors, secrets
FORBIDDEN_DIRS = {"contamination", "generations", "labels", "review", "repaired", "factors"}
FORBIDDEN_NAMES = {"rejected.jsonl", ".env"}
FORBIDDEN_ANYWHERE = "beavertails"


@dataclass
class Upload:
    source: str  # "runs" (the private runs repo) or "local" (this checkout)
    path: str  # the path at the source
    dest: str  # the path in the public repo


@dataclass
class RepoPlan:
    repo_id: str
    repo_type: str  # "dataset" or "model"
    card: str
    uploads: list[Upload] = field(default_factory=list)


def is_forbidden(path: str) -> bool:
    """True for any path that must never go into a public repo."""
    parts = path.lower().split("/")
    return bool(FORBIDDEN_DIRS & set(parts[:-1])) or parts[-1] in FORBIDDEN_NAMES or FORBIDDEN_ANYWHERE in path.lower()


def _under(files: list[str], prefix: str) -> list[str]:
    return sorted(f for f in files if f.startswith(prefix) and not is_forbidden(f))


def data_plan(files: list[str]) -> RepoPlan:
    """data/processed without the forbidden files, at the repo root (harmful/, xstest/, task_t1/, mmlu/)."""
    prefix = "data/processed/"
    uploads = [Upload("runs", f, f[len(prefix) :]) for f in _under(files, prefix)]
    return RepoPlan(f"{OWNER}/safety-pencil-data", "dataset", DATA_CARD, uploads)


def results_plan(files: list[str], local: list[str]) -> RepoPlan:
    """Result JSONs, figures, job records and logs from the runs repo, plus the local-only files."""
    uploads = [
        Upload("runs", f, f)
        for prefix in ("outputs/results/", "outputs/figures/", "runs/")
        for f in _under(files, prefix)
    ]
    uploads += [Upload("local", f, f) for f in local if not is_forbidden(f)]
    uploads.append(Upload("local", "outputs/summary.md", "summary.md"))
    return RepoPlan(f"{OWNER}/safety-pencil-results", "dataset", RESULTS_CARD, uploads)


def model_plan(files: list[str], run: str) -> RepoPlan:
    """The run's adapters at each fraction, without the empty PEFT template READMEs."""
    uploads = []
    for fraction in FRACTIONS:
        prefix = f"outputs/ckpt/{run}/frac{fraction}/"
        uploads += [
            Upload("runs", f, f"frac{fraction}/{f[len(prefix) :]}")
            for f in _under(files, prefix)
            if not f.endswith("/README.md")
        ]
    return RepoPlan(f"{OWNER}/safety-pencil-lora-run{run}", "model", model_card(run), uploads)


def build_plan(files: list[str], local: list[str]) -> list[RepoPlan]:
    """Every public repo with its uploads, from the runs repo's file list and the local-only files present."""
    plans = [data_plan(files), results_plan(files, local)] + [model_plan(files, run) for run in RUNS]
    for plan in plans:
        missing = [u.dest for u in plan.uploads if u.source == "local" and not (REPO_ROOT / u.path).exists()]
        if missing:
            raise SystemExit(f"{plan.repo_id}: local files missing: {missing}")
        if not plan.uploads:
            raise SystemExit(f"{plan.repo_id}: nothing to upload")
    return plans


def sizes(api: HfApi, plans: list[RepoPlan]) -> dict[str, int]:
    """Bytes per runs-repo path (one metadata call), and per local path."""
    paths = [u.path for plan in plans for u in plan.uploads if u.source == "runs"]
    found = {info.path: info.size for info in api.get_paths_info(SOURCE_REPO, paths, repo_type="dataset")}
    found.update({u.path: (REPO_ROOT / u.path).stat().st_size for p in plans for u in p.uploads if u.source == "local"})
    return found


def print_plan(plans: list[RepoPlan], size_of: dict[str, int]) -> None:
    """Every repo, its visibility, its card's first heading and every file with its size."""
    for plan in plans:
        total = sum(size_of.get(u.path, 0) for u in plan.uploads)
        title = next(line for line in plan.card.splitlines() if line.startswith("# "))
        print(
            f"\n== {plan.repo_id} ({plan.repo_type}, public, not gated): {len(plan.uploads) + 1} files, "
            f"{total / 1e6:.1f} MB; card '{title[2:]}'"
        )
        print("   README.md  (card)")
        for u in plan.uploads:
            print(f"   {u.dest}  ({size_of.get(u.path, 0) / 1e6:.2f} MB, from {u.source}:{u.path})")


def publish(api: HfApi, plan: RepoPlan) -> None:
    """Create the public repo, stage its files with the card, and upload them in one commit."""
    for u in plan.uploads:
        if is_forbidden(u.path) or is_forbidden(u.dest):
            raise SystemExit(f"refusing to publish {u.path}")
    api.create_repo(plan.repo_id, repo_type=plan.repo_type, private=False, exist_ok=True)
    with tempfile.TemporaryDirectory() as staging:
        root = Path(staging)
        for u in plan.uploads:
            target = root / u.dest
            target.parent.mkdir(parents=True, exist_ok=True)
            if u.source == "local":
                shutil.copyfile(REPO_ROOT / u.path, target)
            else:
                shutil.copyfile(hf_hub_download(SOURCE_REPO, u.path, repo_type="dataset"), target)
        (root / "README.md").write_text(plan.card)
        api.upload_folder(
            repo_id=plan.repo_id,
            repo_type=plan.repo_type,
            folder_path=staging,
            commit_message="Safety Pencil submission release",
        )
    print(f"published https://huggingface.co/{'datasets/' if plan.repo_type == 'dataset' else ''}{plan.repo_id}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print the plan and write nothing")
    args = parser.parse_args(argv)
    api = HfApi()
    files = api.list_repo_files(SOURCE_REPO, repo_type="dataset")
    plans = build_plan(files, list(LOCAL_ONLY))
    print_plan(plans, sizes(api, plans))
    if args.dry_run:
        print("\ndry run: nothing created or uploaded")
        return 0
    for plan in plans:
        publish(api, plan)
    return 0


if __name__ == "__main__":
    sys.exit(main())
