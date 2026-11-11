"""jobs/launch.py, jobs/sync.py and the stage scripts without the network: which files count as code, what a
job is started with, which paths may never be pushed, how stage chains are written, and which commands each
stage runs (with a fake `python` that records its arguments instead of running anything)."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub.utils import filter_repo_objects

REPO_ROOT = Path(__file__).resolve().parents[1]
JOBS = REPO_ROOT / "jobs"
# records each call and prints $FAKE_STDOUT; runs nothing and always succeeds
FAKE_PYTHON = '#!/bin/sh\necho "$*" >> "$CALLS"\nprintf "%s" "$FAKE_STDOUT"\n'


def run_stage(tmp_path: Path, stage: str, *args: str, stage_arg: str = "", stdout: str = "") -> list[str]:
    """Run jobs/stages/<stage>.sh in tmp_path with the fake `python` first on PATH; returns its calls, one string
    of arguments per call. The scripts it names never run, but its own file operations happen in tmp_path."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    (fake_bin / "python").write_text(FAKE_PYTHON)
    (fake_bin / "python").chmod(0o755)
    calls = tmp_path / "calls.txt"
    calls.write_text("")
    environment = dict(os.environ)
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    environment["CALLS"] = str(calls)
    environment["STAGE_ARG"] = stage_arg
    environment["FAKE_STDOUT"] = stdout
    subprocess.run(["bash", str(JOBS / "stages" / f"{stage}.sh"), *args], cwd=tmp_path, env=environment, check=True)
    return calls.read_text().splitlines()


def scripts_run(calls: list[str]) -> list[str]:
    """The pipeline scripts a stage ran, in order, with a subcommand where there is one (e.g. "06_repair sweep")."""
    steps = []
    for call in calls:
        words = call.split()
        if not words[0].startswith("scripts/"):
            continue
        step = Path(words[0]).stem
        if step == "06_repair":
            step = f"{step} {words[1]}"
        steps.append(step)
    return steps


def out_names(calls: list[str]) -> list[str]:
    """The --out-name of every call that has one, in order."""
    names = []
    for call in calls:
        words = call.split()
        if "--out-name" in words:
            names.append(words[words.index("--out-name") + 1])
    return names


def _load_file(path: Path):
    """Import a script by path (its file name may start with a digit)."""
    module_name = f"loaded_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


def _load(name):
    return _load_file(JOBS / f"{name}.py")


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
    [
        "outputs/generations",
        "outputs/generations/A/en.jsonl",
        "./outputs/labels/",
        ".cache/auditkit",
        "outputs/ckpt/repaired/runA_frac100_full_mu0.1",
        "outputs/factors",
        "outputs/factors/harm/en",
        "outputs/factors/harm/en/layers0-2.safetensors",
    ],
)
def test_push_refuses_generations_labels_checkpoints_and_factor_tensors(sync, path):
    assert sync.is_forbidden(path)
    with pytest.raises(SystemExit):
        sync.push([path])


@pytest.mark.parametrize(
    "path",
    [
        "outputs/results",
        "outputs/ckpt/A",
        "data/processed",
        "outputs/generations_old",
        "outputs/factors/harm/en/meta.json",
        "outputs/factors/task/A_frac100/meta.json",
    ],
)
def test_push_allows_results_adapters_data_and_factor_meta(sync, path):
    assert not sync.is_forbidden(path)


def test_wait_uses_one_deadline_for_all_paths(sync, monkeypatch):
    monkeypatch.setenv("RUNS_REPO", "me/safety-pencil-runs")
    clock = {"now": 0.0}
    checks = []

    def fake_file_exists(repo, path, repo_type):
        checks.append(path)
        return path == "found.json"

    def fake_sleep(seconds):
        clock["now"] += seconds

    fake_time = SimpleNamespace(time=lambda: clock["now"], sleep=fake_sleep)
    monkeypatch.setattr(sync, "file_exists", fake_file_exists)
    monkeypatch.setattr(sync, "time", fake_time)

    assert not sync.wait(["found.json", "missing1.json", "missing2.json"], minutes=3)
    assert clock["now"] <= 4 * 60  # one 3-minute deadline, not one per missing path
    assert checks.count("found.json") == 1  # a found path is not polled again
    assert sync.wait(["found.json"], minutes=3)


def test_pencil_is_refused_on_a_machine_without_the_disk(launch):
    launch.check_disk("pencil", "l4x1")
    launch.check_disk("finetune", "a10g-small")  # small stages fit anywhere
    with pytest.raises(SystemExit, match="pencil needs about 300 GB"):
        launch.check_disk("pencil", "a10g-small")


def test_pencil_stage_runs_every_step_and_frees_each_checkpoint(tmp_path):
    hinglish = tmp_path / "data" / "processed" / "harmful" / "hinglish"
    hinglish.mkdir(parents=True)
    (hinglish / "factor.jsonl").write_text("")
    repaired = tmp_path / "outputs" / "ckpt" / "repaired"
    for mu in ("0.1", "1"):  # what 06_repair.py sweep --save would leave behind
        (repaired / f"runA_frac100_full_mu{mu}").mkdir(parents=True)

    calls = run_stage(tmp_path, "pencil", stage_arg="A", stdout="0.1 1")

    assert scripts_run(calls) == [
        "04_factors",
        "04_factors",
        "04_factors",
        "04_factors",
        "04_leverage",
        "06_repair sweep",
        "02_evaluate",
        "02_evaluate",
        "05_invariance",
        "07_baselines",
        "07_baselines",
        "08_localisation",
        "06_repair forecast",
    ]
    factor_calls = [call for call in calls if call.startswith("scripts/04_factors.py")]
    assert [call.split()[1:5] for call in factor_calls] == [
        ["--side", "harm", "--lang", "en"],
        ["--side", "harm", "--lang", "hi"],
        ["--side", "harm", "--lang", "hinglish"],
        ["--side", "task", "--run", "A"],
    ]
    assert all(call.endswith("--row-sets full") for call in factor_calls)
    sweep = next(call for call in calls if call.startswith("scripts/06_repair.py sweep"))
    assert sweep.endswith("--grad-diag --save 0.1 1")  # the μ grid, read from the config
    assert out_names(calls) == [
        "probe_repair_runA_mu0.1",
        "probe_repair_runA_mu1",
        "baselines_runA_must",
        "baselines_runA_nice",
    ]
    assert not any(repaired.iterdir())  # each 7 GB checkpoint is deleted once evaluated
    wait = next(call for call in calls if call.startswith("jobs/sync.py wait"))
    awaited = [word for word in wait.split() if word.startswith("outputs/results/")]
    assert len(awaited) == 1 + 3 * 4  # θ0, then runs A, C, D × 4 adapters
    assert "outputs/results/probe_runC_frac25.json" in awaited


def test_pencil_stage_without_hinglish_prompts_uses_en_and_hi(tmp_path):
    calls = run_stage(tmp_path, "pencil", stage_arg="A", stdout="0.1")
    harm_langs = [call.split()[4] for call in calls if call.startswith("scripts/04_factors.py --side harm")]
    assert harm_langs == ["en", "hi"]


@pytest.fixture(scope="module")
def ledger():
    return _load("ledger")


@pytest.fixture(scope="module")
def report():
    return _load("report")


def test_ledger_record_starts_and_finishes(ledger, tmp_path):
    environment = {"STAGE": "pencil", "STAGE_ARG": "A", "PENCIL_GIT_HASH": "abc1234", "GPU_FLAVOR": "l4x1"}
    record = ledger.new_record("20261112T000000Z-pencil-A", environment, now=0.0)
    assert record["stage"] == "pencil" and record["arg"] == "A" and record["code"] == "abc1234"
    assert record["started"] == "1970-01-01T00:00:00Z" and record["status"] == "running"

    results = tmp_path / "results"
    results.mkdir()
    (results / "mine.json").write_text('{"run_id": "20261112T000000Z-pencil-A"}')
    (results / "pulled.json").write_text('{"run_id": "20261111T120000Z-finetune-A"}')
    (results / "older.json").write_text('{"name": "from before the ledger"}')
    mine = ledger.results_of("20261112T000000Z-pencil-A", results)
    assert [path.name for path in mine] == ["mine.json"]  # pulled and older results belong to other runs

    finished = ledger.finish_record(record, exit_status=1, results=mine, now=90 * 60)
    assert finished["status"] == "failed (exit 1)"
    assert finished["minutes"] == 90.0 and finished["results"] == ["mine"]
    assert ledger.finish_record(record, exit_status=0, results=[], now=60)["status"] == "succeeded"


def test_ledger_start_then_finish_uploads_record_log_and_result_copies(ledger, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STAGE", "evaluate")
    monkeypatch.setenv("STAGE_ARG", "theta0")
    uploads = []

    def fake_upload(folder, run_id, message):
        uploads.append(sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()))

    monkeypatch.setattr(ledger, "_upload", fake_upload)
    run_id = "20261112T000000Z-evaluate-theta0"
    ledger.start(run_id)
    (tmp_path / "outputs" / "results").mkdir(parents=True)
    (tmp_path / "outputs" / "results" / "probe_theta0.json").write_text(json.dumps({"run_id": run_id}))
    (tmp_path / "outputs" / "runs" / run_id / "log.txt").write_text("== stage evaluate theta0\n")
    ledger.finish(run_id, exit_status=0)

    assert uploads == [["run.json"], ["log.txt", "results/probe_theta0.json", "run.json"]]
    record = json.loads((tmp_path / "outputs" / "runs" / run_id / "run.json").read_text())
    assert record["status"] == "succeeded" and record["results"] == ["probe_theta0"]


# creates the venv, accepts installs, and fails every `uv run` (the ledger) as if the Hub were down
FAKE_UV = """#!/bin/sh
case "$1" in
  venv) mkdir -p .venv/bin && : > .venv/bin/activate ;;
  run) echo "ledger call: $*" >> "$CALLS"; exit 1 ;;
esac
exit 0
"""


def run_entrypoint(tmp_path: Path, stage_exit: int) -> subprocess.CompletedProcess:
    """Run a copy of jobs/entrypoint.sh around a stub stage that exits with `stage_exit`, with a fake `uv` whose
    ledger calls all fail and the fake `python`; returns the finished process."""
    tree = tmp_path / f"exit{stage_exit}"
    (tree / "jobs" / "stages").mkdir(parents=True)
    (tree / "scripts").mkdir()
    shutil.copy(JOBS / "entrypoint.sh", tree / "jobs" / "entrypoint.sh")
    (tree / "jobs" / "stages" / "stub.sh").write_text(f'echo "stub stage ran"\nexit {stage_exit}\n')
    (tree / "scripts" / "setup_lexsi.sh").write_text("exit 0\n")
    fake_bin = tree / "bin"
    fake_bin.mkdir()
    for name, script in (("uv", FAKE_UV), ("python", FAKE_PYTHON)):
        (fake_bin / name).write_text(script)
        (fake_bin / name).chmod(0o755)
    environment = dict(os.environ)
    environment.update(
        PATH=f"{fake_bin}{os.pathsep}{environment['PATH']}",
        CALLS=str(tree / "calls.txt"),
        FAKE_STDOUT="",
        STAGE="stub",
        STAGE_ARG="",
        THEN="",
        HUB_CLIENT="huggingface_hub",
        PENCIL_GIT_HASH="abc1234",
    )
    command = ["bash", str(tree / "jobs" / "entrypoint.sh")]
    return subprocess.run(command, env=environment, capture_output=True, text=True)


@pytest.mark.parametrize("stage_exit", [0, 3])
def test_a_failing_ledger_never_stops_a_job_or_changes_its_status(tmp_path, stage_exit):
    finished = run_entrypoint(tmp_path, stage_exit)
    calls = (tmp_path / f"exit{stage_exit}" / "calls.txt").read_text()
    log = next((tmp_path / f"exit{stage_exit}" / "outputs" / "runs").glob("*/log.txt")).read_text()
    assert finished.returncode == stage_exit  # the stage's own status, though both ledger calls failed
    assert "ledger.py start" in calls and f"ledger.py finish {stage_exit}" in calls
    assert "stub stage ran" in log
    assert "ledger: start not recorded" in log


def test_ledger_log_squeezes_progress_bars_and_masks_tokens(ledger):
    text = "step 1\n 10%|█\r 50%|███\r100%|█████\nlogin with hf_" + "a" * 30 + "\n"
    assert ledger.clean_log(text) == "step 1\n100%|█████\nlogin with hf_***\n"


def _probe_result(refusals: dict[str, list[str]]) -> dict:
    """A probe result with harmful items labelled as given, per language, and matching rates."""
    items = []
    metrics = {}
    for lang, labels in refusals.items():
        for index, label in enumerate(labels):
            items.append({"id": f"{lang}-{index}", "lang": lang, "refusal": label, "unsafe": False})
        k = labels.count("REFUSE")
        metrics[lang] = {"harmful": {"refusal": {"k": k, "n": len(labels), "p": k / len(labels), "lo": 0, "hi": 1}}}
    return {"items": {"harmful": items}, "metrics": metrics, "task": {"nll_per_token": 2.0}}


def test_gate1_passes_on_a_large_paired_drop_only(report):
    theta0 = _probe_result({"en": ["REFUSE"] * 20, "hi": ["REFUSE"] * 10 + ["COMPLY"] * 10})
    after = _probe_result({"en": ["COMPLY"] * 20, "hi": ["REFUSE"] * 10 + ["PARTIAL"] * 10})
    rows = {row["lang"]: row for row in report.gate1(theta0, after)}
    assert rows["en"]["drop"] == 1.0 and rows["en"]["p"] < 0.001 and rows["en"]["passes"]
    assert rows["hi"]["drop"] == 0.0 and not rows["hi"]["passes"]  # PARTIAL is not a refusal, but neither was COMPLY
    assert rows["en"]["p_holm"] >= rows["en"]["p"]


def test_summary_page_has_the_ledger_gate1_and_probe_tables(report):
    run = {"run_id": "r1", "stage": "finetune", "arg": "A", "code": "abc", "machine": "l4x1"}
    run.update(started="2026-11-11T16:47:00Z", status="succeeded", minutes=80.0, results=["probe_runA_frac100"])
    runs = [run]
    theta0 = _probe_result({"en": ["REFUSE"] * 4 + ["COMPLY"] * 4})
    final = _probe_result({"en": ["COMPLY"] * 8})
    results = {
        "probe_theta0": {"result": theta0, "run_id": None},
        "probe_runA_frac100": {"result": final, "run_id": "r1"},
        "leverage_en_runA": {"result": {}, "run_id": "r2", "timestamp": "t", "git": "abc"},
    }
    page = report.summary(runs, results)
    assert "| 2026-11-11T16:47:00Z | r1 | finetune A | abc | l4x1 | succeeded | 80.0 | 1 |" in page
    assert "| A | en | 8 | 50% | 0% | +50% |" in page  # drop = refusal at θ0 minus after
    assert "| theta0 | en | 50% [0%, 100%] (4/8) |" in page
    assert "| leverage_en_runA | t | r2 | abc |" in page


def test_baselines_out_name_defaults_to_the_run():
    parser = _load_file(REPO_ROOT / "scripts" / "07_baselines.py").build_parser()
    assert parser.parse_args(["--run", "A"]).out_name is None  # main() then writes baselines_runA
    assert parser.parse_args(["--out-name", "baselines_runA_nice"]).out_name == "baselines_runA_nice"


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
