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
# Records each call and prints $FAKE_STDOUT. Runs nothing; succeeds unless the call contains $FAKE_FAIL.
FAKE_PYTHON = """#!/bin/sh
echo "$*" >> "$CALLS"
printf "%s" "$FAKE_STDOUT"
if [ -n "$FAKE_FAIL" ]; then
  case "$*" in *"$FAKE_FAIL"*) exit 1 ;; esac
fi
exit 0
"""


def run_stage(
    tmp_path: Path,
    stage: str,
    *args: str,
    stage_arg: str = "",
    stdout: str = "",
    fail: str = "",
    tier: str = "",
    expect_exit: int = 0,
) -> list[str]:
    """Run jobs/stages/<stage>.sh in tmp_path with the fake `python` first on PATH; returns its calls, one string
    of arguments per call. The scripts it names never run (a call containing `fail` fails), but the stage's own
    file operations happen in tmp_path. Checks the stage exits with `expect_exit`."""
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
    environment["FAKE_FAIL"] = fail
    environment["EVAL_TIER"] = tier
    command = ["bash", str(JOBS / "stages" / f"{stage}.sh"), *args]
    finished = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert finished.returncode == expect_exit, finished.stdout + finished.stderr
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


def test_evaluate_stage_at_the_full_tier_uses_the_test_split_and_the_final_adapter(tmp_path):
    run_calls = run_stage(tmp_path, "evaluate", "A", tier="full")
    assert out_names(run_calls) == ["test_runA_frac100"]
    assert all("--tier full" in call for call in run_calls if call.startswith("scripts/02_evaluate.py"))
    assert out_names(run_stage(tmp_path, "evaluate", "theta0", tier="full")) == ["test_theta0"]


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
        "05_invariance",
        "06_repair forecast",
        "06_repair sweep",
        "02_evaluate",
        "02_evaluate",
        "08_localisation",
    ]  # no baselines: they run in the compare stage
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
    ]
    assert not any(repaired.iterdir())  # each 7 GB checkpoint is deleted once evaluated
    wait = next(call for call in calls if call.startswith("jobs/sync.py wait"))
    awaited = [word for word in wait.split() if word.startswith("outputs/results/")]
    assert len(awaited) == 1 + 3 * 4  # θ0, then runs A, C, D × 4 adapters
    assert "outputs/results/probe_runC_frac25.json" in awaited


def test_a_failing_step_never_stops_the_later_steps_but_fails_the_stage(tmp_path):
    """The crash that lost pencil D's factors: now leverage fails, every later step still runs, exit 1."""
    calls = run_stage(tmp_path, "pencil", stage_arg="A", stdout="0.1", fail="04_leverage", expect_exit=1)
    steps = scripts_run(calls)
    assert steps[steps.index("04_leverage") + 1 :] == [
        "05_invariance",
        "06_repair forecast",
        "06_repair sweep",
        "08_localisation",
    ]


def test_a_failing_factor_pass_stops_the_stage(tmp_path):
    """Nothing can run without the factors, so a failed pass ends the stage at once."""
    calls = run_stage(tmp_path, "pencil", stage_arg="A", stdout="0.1", fail="--side task", expect_exit=1)
    assert "04_leverage" not in scripts_run(calls)


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


def _probe_result(refusals: dict[str, list[str]], unsafe: dict[str, list[bool]] | None = None) -> dict:
    """A probe result with harmful items labelled as given per language (unsafe defaults to all False), and
    matching refusal rates."""
    items = []
    metrics = {}
    for lang, labels in refusals.items():
        unsafe_labels = [False] * len(labels)
        if unsafe and lang in unsafe:
            unsafe_labels = unsafe[lang]
        for index, (label, harmful) in enumerate(zip(labels, unsafe_labels)):
            items.append({"id": f"{lang}-{index}", "lang": lang, "refusal": label, "unsafe": harmful})
        k = labels.count("REFUSE")
        metrics[lang] = {"harmful": {"refusal": {"k": k, "n": len(labels), "p": k / len(labels), "lo": 0, "hi": 1}}}
    return {"items": {"harmful": items}, "metrics": metrics, "task": {"nll_per_token": 2.0}}


def test_gate1_passes_on_a_large_paired_refusal_drop_only(report):
    theta0 = _probe_result({"en": ["REFUSE"] * 20, "hi": ["REFUSE"] * 10 + ["COMPLY"] * 10})
    after = _probe_result({"en": ["COMPLY"] * 20, "hi": ["REFUSE"] * 10 + ["PARTIAL"] * 10})
    rows = {row["lang"]: row for row in report.gate1(theta0, after)}
    assert rows["en"]["refusal"]["change"] == -1.0 and rows["en"]["refusal"]["p"] < 0.001 and rows["en"]["passes"]
    assert rows["hi"]["refusal"]["change"] == 0.0 and not rows["hi"]["passes"]  # PARTIAL is no refusal; nor COMPLY
    assert rows["en"]["refusal"]["p_holm"] >= rows["en"]["refusal"]["p"]


def test_gate1_reports_the_paired_change_in_harmful_compliance(report):
    theta0 = _probe_result({"en": ["COMPLY"] * 20})
    after = _probe_result({"en": ["COMPLY"] * 20}, unsafe={"en": [True] * 15 + [False] * 5})
    unsafe = report.gate1(theta0, after)[0]["unsafe"]
    assert (unsafe["before"], unsafe["after"], unsafe["change"]) == (0.0, 0.75, 0.75)
    assert unsafe["p"] < 0.001  # 15 prompts turned unsafe, none the other way


def test_p2_ranks_the_forecast_against_s_unsafe_and_refusal(report):
    fracs = [0.25, 0.5, 0.75, 1.0]
    theta0 = _probe_result({"en": ["REFUSE"] * 5 + ["COMPLY"] * 5})
    theta0["metrics"]["en"]["harmful"]["unsafe"] = {"k": 1, "n": 10, "p": 0.1}
    results = {"probe_theta0": {"result": theta0}}
    forecast_rows = []
    drift_rows = []
    for index, frac in enumerate(fracs):
        forecast_rows.append({"run": "D", "frac": frac, "lang": "en", "forecast": float(index)})
        drift_rows.append({"run": "D", "frac": frac, "lang": "en", "S": 0.01 * (index + 1)})  # same order
        probe = _probe_result({"en": ["REFUSE"] * 6 + ["COMPLY"] * 4})  # refusal rises: no drop to rank
        probe["metrics"]["en"]["harmful"]["unsafe"] = {"k": 2 + index, "n": 10, "p": 0.1 * (2 + index)}
        results[f"probe_runD_frac{round(frac * 100)}"] = {"result": probe}
    results["forecast_p2"] = {"result": {"rows": forecast_rows}}
    results["drift_S"] = {"result": {"rows": drift_rows}}

    rows = report.p2_rows(results)
    assert [row["S"] for row in rows] == [0.01, 0.02, 0.03, 0.04]
    assert [round(row["unsafe_change"], 2) for row in rows] == [0.1, 0.2, 0.3, 0.4]
    assert all(round(row["refusal_drop"], 2) == -0.1 for row in rows)
    page = "\n".join(report.p2_section(results))
    assert "| measured drift S on dev | 4 | 1.00 [" in page
    assert "| change in harmful compliance | 4 | 1.00 [" in page


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
    # refusal falls 50 points but on 4 discordant prompts only (p = 0.125), so the gate fails
    assert "| A | en | 8 | 50% → 0% | -50 | 0.125 | 0.125 | 0% → 0% | +0 | 1 | 1 | fail |" in page
    assert "| theta0 | en | 50% [0%, 100%] (4/8) |" in page
    assert "| leverage_en_runA | t | r2 | abc |" in page


def test_drift_stage_measures_every_runs_adapters(tmp_path):
    calls = run_stage(tmp_path, "drift")
    assert scripts_run(calls) == ["06_repair drift"]
    pull = calls[0]
    assert pull.startswith("jobs/sync.py pull")
    assert all(f"outputs/ckpt/{run}/**" in pull for run in ("A", "C", "D"))
    assert calls[-1] == "jobs/sync.py push outputs/results"


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


def test_tier_reaches_the_evaluate_job_only(launch, monkeypatch):
    calls = []

    def fake_run_job(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(id="job1", url="https://hf.co/jobs/job1")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "run_job", fake_run_job)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_secret")
    launch.start_job("evaluate", "theta0", "me/code", "me/runs", "abc1234", "l4x1", "full")
    launch.start_job("evaluate", "A", "me/code", "me/runs", "abc1234")
    full, probe = calls
    assert full["env"]["EVAL_TIER"] == "full" and full["name"] == "pencil-evaluate-theta0-full"
    assert "EVAL_TIER" not in probe["env"]  # the stage defaults to probe
    with pytest.raises(SystemExit):
        launch.parse_args(["pencil", "A", "--tier", "full"])


def _compare_checkpoints(tmp_path: Path) -> tuple[Path, Path]:
    """The folders the compare stage's sweep and finalist baselines would save, under tmp_path."""
    repaired = tmp_path / "outputs" / "ckpt" / "repaired"
    saved_baselines = tmp_path / "outputs" / "ckpt" / "baselines"
    for mu in ("0.1", "1"):
        (repaired / f"runA_frac100_full_mu{mu}").mkdir(parents=True)
    for name in ("runA_diag_fisher_0.1", "runA_regmean_3.16228"):
        (saved_baselines / name).mkdir(parents=True)
    return repaired, saved_baselines


def test_compare_stage_puts_the_finalists_before_p3_and_localisation(tmp_path):
    repaired, saved_baselines = _compare_checkpoints(tmp_path)
    calls = run_stage(tmp_path, "compare", stage_arg="A", stdout="0.1")  # the fake 09_finalists says μ* = 0.1

    steps = scripts_run(calls)
    assert steps[:6] == [
        "04_factors",
        "04_factors",
        "04_factors",
        "06_repair forecast",
        "06_repair sweep",
        "07_baselines",
    ]
    # P1 after P6, then the other μ (P3; 09_finalists names μ* so it is skipped), then P5
    assert steps[-4:] == ["05_invariance", "09_finalists", "02_evaluate", "08_localisation"]
    # localisation is last: running out of time costs it, not the baseline comparison
    sweep = next(call for call in calls if call.startswith("scripts/06_repair.py sweep"))
    assert "--save" in sweep and "--grad-diag" not in sweep and sweep.endswith("--out-name repair_sweep_compare_runA")
    assert out_names(calls) == [
        "repair_sweep_compare_runA",
        "baselines_runA_must",
        "probe_repair_runA_mu0.1",  # pencil μ*
        "baselines_runA_finalists",
        "probe_finalist_runA_diag_fisher_0.1",
        "probe_finalist_runA_regmean_3.16228",
        "probe_repair_runA_mu1",  # the other μ, for P3
    ]
    assert all("--tier probe" in call for call in calls if call.startswith("scripts/02_evaluate.py"))
    # no control file: no test tier, and only μ* and RegMean's finalist were kept for it
    assert [path.name for path in repaired.iterdir()] == ["runA_frac100_full_mu0.1"]
    assert [path.name for path in saved_baselines.iterdir()] == ["runA_regmean_3.16228"]


def test_compare_stage_test_evaluates_what_the_control_file_asks_for(tmp_path):
    _compare_checkpoints(tmp_path)
    control = tmp_path / "outputs" / "control"
    control.mkdir(parents=True)
    (control / "compare_test_tier.txt").write_text("pencil regmean\n")

    calls = run_stage(tmp_path, "compare", stage_arg="A", stdout="0.1")
    assert out_names(calls)[-2:] == ["test_repair_runA_mu0.1", "test_finalist_runA_regmean_3.16228"]
    test_tier = [call for call in calls if "--tier full" in call]
    assert len(test_tier) == 2


def test_compare_is_refused_without_the_disk(launch):
    with pytest.raises(SystemExit, match="compare needs about 300 GB"):
        launch.check_disk("compare", "a10g-small")
