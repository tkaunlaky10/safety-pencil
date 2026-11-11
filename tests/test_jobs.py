"""jobs/launch.py, jobs/sync.py and the stage scripts without the network: which files count as code, what a
job is started with, which paths may never be pushed, how stage chains are written, and which commands each
stage runs (with a fake `python` that records its arguments instead of running anything)."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub.utils import filter_repo_objects

JOBS = Path(__file__).resolve().parents[1] / "jobs"
FAKE_PYTHON = '#!/bin/sh\necho "$*" >> "$CALLS"\n'  # records each call, runs nothing, always succeeds


def run_stage(tmp_path: Path, stage: str, *args: str) -> list[str]:
    """Run jobs/stages/<stage>.sh from the repo root with the fake `python` first on PATH; returns its calls,
    one string of arguments per call."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    (fake_bin / "python").write_text(FAKE_PYTHON)
    (fake_bin / "python").chmod(0o755)
    calls = tmp_path / "calls.txt"
    calls.write_text("")
    environment = dict(os.environ)
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    environment["CALLS"] = str(calls)
    environment["STAGE_ARG"] = ""
    subprocess.run(["bash", str(JOBS / "stages" / f"{stage}.sh"), *args], cwd=JOBS.parent, env=environment, check=True)
    return calls.read_text().splitlines()


def out_names(calls: list[str]) -> list[str]:
    """The --out-name of every call that has one, in order."""
    names = []
    for call in calls:
        words = call.split()
        if "--out-name" in words:
            names.append(words[words.index("--out-name") + 1])
    return names


def _load(name):
    module_name = f"jobs_{name}"
    spec = importlib.util.spec_from_file_location(module_name, JOBS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def launch():
    return _load("launch")


@pytest.fixture(scope="module")
def sync():
    return _load("sync")


def test_code_upload_takes_code_and_leaves_environment_and_outputs(launch):
    files = [
        "src/pencil/repair.py",
        "src/pencil/__pycache__/repair.cpython-312.pyc",
        "scripts/03_finetune.py",
        "jobs/stages/calibrate.sh",
        "configs/experiment.yaml",
        "tests/test_math.py",
        "patches/SafeTune.patch",
        "pyproject.toml",
        "requirements-main.lock.txt",
        "README.md",
        ".env",
        ".venv/lib/python3.12/site-packages/torch/README.md",
        "outputs/results/probe_theta0.json",
        "data/processed/harmful/en/test.jsonl",
        "third_party/SafeTune/setup.py",
        "verification/VERIFICATION.md",
    ]
    kept = set(filter_repo_objects(files, allow_patterns=launch.CODE_FILES, ignore_patterns=launch.NOT_CODE))
    assert kept == {
        "src/pencil/repair.py",
        "scripts/03_finetune.py",
        "jobs/stages/calibrate.sh",
        "configs/experiment.yaml",
        "tests/test_math.py",
        "patches/SafeTune.patch",
        "pyproject.toml",
        "requirements-main.lock.txt",
        "README.md",
    }


def test_every_stage_has_a_script_and_chains_name_real_stages(launch):
    for name, stage in launch.STAGES.items():
        assert (JOBS / "stages" / f"{name}.sh").is_file()
        for follower in stage.then:
            assert follower.split(":")[0] in launch.STAGES


def test_start_job_passes_stage_code_and_token(launch, monkeypatch):
    calls = {}

    def fake_run_job(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(id="job1", url="https://hf.co/jobs/job1")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "run_job", fake_run_job)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_secret")
    launch.start_job("finetune", "A", "me/safety-pencil-code", "me/safety-pencil-runs", "abc1234")

    assert calls["flavor"] == "l4x1"
    assert calls["secrets"] == {"HF_TOKEN": "hf_secret"}
    assert calls["env"]["STAGE"] == "finetune"
    assert calls["env"]["STAGE_ARG"] == "A"
    assert calls["env"]["PENCIL_GIT_HASH"] == "abc1234"
    assert calls["env"]["RUNS_REPO"] == "me/safety-pencil-runs"
    assert "hf_secret" not in str(calls["env"])  # the token travels only as a secret
    assert calls["command"][:2] == ["bash", "-c"]
    assert calls["labels"] == {"project": "safety-pencil", "stage": "finetune", "arg": "A"}
    assert calls["name"] == "pencil-finetune-A"

    launch.start_job("calibrate", "", "me/safety-pencil-code", "me/safety-pencil-runs", "abc1234")
    assert calls["name"] == "pencil-calibrate"
    assert calls["labels"] == {"project": "safety-pencil", "stage": "calibrate"}


def test_gpu_choice_reaches_gpu_stages_only_and_is_passed_down(launch, monkeypatch):
    calls = []

    def fake_run_job(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(id="job1", url="https://hf.co/jobs/job1")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "run_job", fake_run_job)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_secret")
    launch.start_job("finetune", "A", "me/code", "me/runs", "abc1234", "a10g-small")
    launch.start_job("hinglish", "", "me/code", "me/runs", "abc1234", "a10g-small")

    finetune, hinglish = calls
    assert finetune["flavor"] == "a10g-small"
    assert hinglish["flavor"] == "cpu-basic"
    assert hinglish["env"]["GPU_FLAVOR"] == "a10g-small"  # hinglish starts evaluate theta0 on the same GPU


def test_gpu_option_allows_24gb_cards_only_and_chained_launches_inherit_it(launch, monkeypatch):
    with pytest.raises(SystemExit):
        launch.parse_args(["calibrate", "--gpu", "a100-large"])
    assert launch.chosen_gpu(launch.parse_args(["calibrate"])) == "l4x1"
    assert launch.chosen_gpu(launch.parse_args(["calibrate", "--gpu", "a10g-small"])) == "a10g-small"

    monkeypatch.setenv("GPU_FLAVOR", "a10g-small")
    assert launch.chosen_gpu(launch.parse_args(["finetune", "A", "--chained"])) == "a10g-small"
    assert launch.chosen_gpu(launch.parse_args(["finetune", "A"])) == "l4x1"  # the variable counts only when chained


def test_evaluate_stage_names_results_the_way_the_forecast_reads_them(tmp_path):
    """06_repair.py forecast reads probe_theta0 and probe_run{R}_frac{F}."""
    run_calls = run_stage(tmp_path, "evaluate", "A")
    assert out_names(run_calls) == ["probe_runA_frac25", "probe_runA_frac50", "probe_runA_frac75", "probe_runA_frac100"]
    evaluations = [call for call in run_calls if call.startswith("scripts/02_evaluate.py")]
    assert all("--langs en hi hinglish" in call for call in evaluations)  # the fake `wait` finds the Hinglish data
    assert out_names(run_stage(tmp_path, "evaluate", "theta0")) == ["probe_theta0"]


def test_finetune_and_evaluate_need_an_argument(launch):
    with pytest.raises(SystemExit):
        launch.parse_args(["finetune"])
    assert launch.parse_args(["evaluate", "theta0"]).arg == "theta0"
    assert launch.parse_args(["calibrate"]).arg == ""


@pytest.mark.parametrize(
    "path",
    ["outputs/generations", "outputs/generations/A/en.jsonl", "./outputs/labels/", ".cache/auditkit"],
)
def test_push_refuses_generations_and_label_sheets(sync, path):
    assert sync.is_forbidden(path)
    with pytest.raises(SystemExit):
        sync.push([path])


@pytest.mark.parametrize("path", ["outputs/results", "outputs/ckpt/A", "data/processed", "outputs/generations_old"])
def test_push_allows_results_adapters_and_data(sync, path):
    assert not sync.is_forbidden(path)


def test_push_uploads_files_and_folders_at_the_same_path(sync, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RUNS_REPO", "me/safety-pencil-runs")
    (tmp_path / "outputs" / "results").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "model_facts.json").write_text("{}")
    uploads = []

    def fake_upload_folder(**kwargs):
        uploads.append(("folder", kwargs["path_in_repo"]))

    def fake_upload_file(**kwargs):
        uploads.append(("file", kwargs["path_in_repo"]))

    monkeypatch.setattr(sync, "upload_folder", fake_upload_folder)
    monkeypatch.setattr(sync, "upload_file", fake_upload_file)

    sync.push(["outputs/results", "configs/model_facts.json", "outputs/ckpt/A"])  # the last one doesn't exist
    assert uploads == [("folder", "outputs/results"), ("file", "configs/model_facts.json")]
