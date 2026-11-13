"""The finalist rule (pencil.finalists), 09_pick_test_settings.py on small result files, and 07_baselines.py
--finalists grid restriction."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from pencil.finalists import closest_setting, pencil_mu_star, task_nll_limit

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
NLL_THETA0, NLL_THETA1 = 2.0, 1.0  # the fine-tune gained 1.0; keeping 90% allows NLL up to 1.1


def _load_script(name: str):
    module_name = f"script_{name}"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


def _points(nll_by_mu: dict, failed=()) -> list[dict]:
    """Sweep points with the given dev task NLL per μ; μ in `failed` failed their trust-region check."""
    return [{"mu": mu, "task_nll": nll, "trust_region_failed": mu in failed} for mu, nll in nll_by_mu.items()]


def test_mu_star_is_the_strongest_repair_that_keeps_90_percent_of_the_task_gain():
    assert task_nll_limit(NLL_THETA0, NLL_THETA1) == pytest.approx(1.1)
    points = _points({0.01: 1.6, 0.03: 1.3, 0.1: 1.08, 0.3: 1.03, 1.0: 1.01})
    assert pencil_mu_star(points, NLL_THETA0, NLL_THETA1)["mu"] == 0.1


def test_mu_star_skips_failed_trust_regions_and_errors():
    points = _points({0.1: 1.05, 0.3: 1.03}, failed={0.1}) + [{"error": "RuntimeError: ..."}]
    mu_star = pencil_mu_star(points, NLL_THETA0, NLL_THETA1)
    assert mu_star["mu"] == 0.3 and mu_star["backup"] is False


def test_no_mu_within_the_limit_gives_the_backup_mu_star_with_the_lowest_nll():
    """No μ*: no accepted μ keeps 90% of the gain (limit 1.1). The backup is the accepted μ with the lowest
    dev NLL (0.3 at 1.2), not the smallest μ, and never a failed one (1.0 at 1.15 failed rule 11)."""
    points = _points({0.03: 1.5, 0.3: 1.2, 1.0: 1.15}, failed={1.0})
    mu_star = pencil_mu_star(points, NLL_THETA0, NLL_THETA1)
    assert mu_star == {"mu": 0.3, "task_nll": 1.2, "limit": pytest.approx(1.1), "backup": True}


def test_no_accepted_mu_means_no_mu_star_at_all():
    from pencil.finalists import NoAcceptedMu

    with pytest.raises(NoAcceptedMu, match="no μ passed rule 11"):
        pencil_mu_star(_points({0.1: 1.05, 0.3: 1.03}, failed={0.1, 0.3}), NLL_THETA0, NLL_THETA1)


def test_baseline_finalist_is_the_setting_closest_in_task_nll():
    rows = [{"setting": 0.3, "task_nll": 1.4}, {"error": "boom"}, {"setting": 3.0, "task_nll": 1.09}]
    assert closest_setting(rows, target_nll=1.08)["setting"] == 3.0


def test_finalists_script_writes_the_picks_and_prints_them_for_the_shell(tmp_path, monkeypatch, capsys):
    finalists = _load_script("09_pick_test_settings")
    results = tmp_path / "results"
    results.mkdir()
    sweep = {"task_nll": {"theta0": NLL_THETA0, "theta1": NLL_THETA1}, "points": _points({0.03: 1.3, 0.1: 1.08})}
    baselines = {}
    for method, nll_by_setting in (
        ("regmean", {0.3: 1.5, 3.0: 1.07}),
        ("diag_fisher", {0.01: 1.2, 0.1: 1.09}),
        ("wiseft", {0.25: 1.6, 0.75: 1.1}),
    ):
        rows = [{"setting": setting, "task_nll": nll, "S_mean": 0.01} for setting, nll in nll_by_setting.items()]
        baselines[method] = rows
    (results / "sweep.json").write_text(json.dumps({"result": sweep}))
    (results / "base.json").write_text(json.dumps({"result": baselines}))
    monkeypatch.setattr(finalists, "outputs_dir", lambda name: tmp_path / name)

    common = ["--run", "A", "--sweep", "sweep", "--baselines", "base", "--test-tier", "regmean", "diag_fisher"]
    finalists.main([*common, "--print", "mu"])
    assert capsys.readouterr().out.strip() == "0.1"
    finalists.main([*common, "--print", "baselines"])
    assert capsys.readouterr().out.strip() == "regmean=3.0 diag_fisher=0.1"  # WiSE-FT picked, not evaluated
    finalists.main(common)
    written = json.loads((results / "finalists_runA.json").read_text())["result"]
    assert written["baselines"]["wiseft"]["setting"] == 0.75 and written["test_tier"] == ["regmean", "diag_fisher"]


def test_baselines_finalists_option_runs_exactly_the_picked_grid_value():
    baselines = _load_script("07_baselines")
    grid = baselines.REGMEAN_MUS
    picked = float(repr(grid[5]))  # the value travels through the shell as text
    assert baselines.restrict_grid(grid, picked, "regmean") == [grid[5]]
    assert baselines.restrict_grid(grid, None, "regmean") == grid
    with pytest.raises(ValueError, match="not in its grid"):
        baselines.restrict_grid(grid, 123.0, "regmean")
    assert baselines.parse_finalists(["regmean=3.0", "diag_fisher=0.1"]) == {"regmean": 3.0, "diag_fisher": 0.1}


def test_a_baseline_that_failed_on_dev_is_recorded_as_unavailable_not_fatal(tmp_path, monkeypatch, capsys):
    finalists = _load_script("09_pick_test_settings")
    results = tmp_path / "results"
    results.mkdir()
    sweep = {"task_nll": {"theta0": NLL_THETA0, "theta1": NLL_THETA1}, "points": _points({0.1: 1.08})}
    baselines = {
        "regmean": [{"setting": 3.0, "task_nll": 1.07, "S_mean": 0.01}],
        "diag_fisher": [{"setting": 0.1, "task_nll": 1.09, "S_mean": 0.01}],
        "wiseft": [{"error": "OutOfMemoryError: ..."}],  # e.g. its reference models failed to load
    }
    (results / "sweep.json").write_text(json.dumps({"result": sweep}))
    (results / "base.json").write_text(json.dumps({"result": baselines}))
    monkeypatch.setattr(finalists, "outputs_dir", lambda name: tmp_path / name)

    common = ["--run", "A", "--sweep", "sweep", "--baselines", "base"]
    finalists.main([*common, "--print", "baselines"])
    assert capsys.readouterr().out.strip() == "regmean=3.0 diag_fisher=0.1"  # the broken one is left out
    finalists.main(common)
    written = json.loads((results / "finalists_runA.json").read_text())["result"]
    assert "unavailable" in written["baselines"]["wiseft"]


def test_any_passing_tells_the_stage_whether_rule_11_accepted_a_mu(tmp_path, monkeypatch):
    finalists = _load_script("09_pick_test_settings")
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(finalists, "outputs_dir", lambda name: tmp_path / name)
    assert finalists.main(["--sweep", "missing", "--any-passing"]) == 1  # the sweep wrote nothing
    rejected = {"points": _points({0.1: 1.05, 0.3: 1.03}, failed={0.1, 0.3})}
    (results / "rejected.json").write_text(json.dumps({"result": rejected}))
    assert finalists.main(["--sweep", "rejected", "--any-passing"]) == 1
    some = {"points": _points({0.1: 1.05, 0.3: 1.03}, failed={0.1})}
    (results / "some.json").write_text(json.dumps({"result": some}))
    assert finalists.main(["--sweep", "some", "--any-passing"]) == 0


def test_matching_steps_toward_the_target_nll_inside_the_bracket():
    from pencil.finalists import MATCH_TOLERANCE, next_setting_for_nll

    rows = [{"setting": 0.1, "task_nll": 1.9}, {"setting": 1.0, "task_nll": 1.8}, {"setting": 10.0, "task_nll": 1.75}]
    # log scale: 1.85 is half way between 0.1 and 1.0 in NLL, so half way in log μ
    assert next_setting_for_nll(rows, 1.85, log_scale=True) == pytest.approx(0.1**0.5)
    assert next_setting_for_nll(rows, 1.85, log_scale=False) == pytest.approx(0.55)
    assert next_setting_for_nll(rows, 2.5, log_scale=True) is None  # outside the measured range
    assert next_setting_for_nll(rows, 1.8 + MATCH_TOLERANCE / 2, log_scale=True) is None  # already matched
    assert next_setting_for_nll(rows + [{"error": "boom"}], 1.85, log_scale=True) == pytest.approx(0.1**0.5)


def test_finalists_script_prints_the_target_nll_for_matching(tmp_path, monkeypatch, capsys):
    finalists = _load_script("09_pick_test_settings")
    results = tmp_path / "results"
    results.mkdir()
    sweep = {"task_nll": {"theta0": NLL_THETA0, "theta1": NLL_THETA1}, "points": _points({0.03: 1.3, 0.1: 1.08})}
    (results / "sweep.json").write_text(json.dumps({"result": sweep}))
    (results / "base.json").write_text(json.dumps({"result": {}}))
    monkeypatch.setattr(finalists, "outputs_dir", lambda name: tmp_path / name)
    finalists.main(["--run", "A", "--sweep", "sweep", "--baselines", "base", "--print", "target_nll"])
    assert float(capsys.readouterr().out) == 1.08


class _StubRunner:
    """BaselineRunner without models: each applied setting is "measured" with a made-up NLL and drift."""

    def __init__(self, ctx, run, save_tags, finalists=None):
        self.finalists = finalists or {}
        self.pencil_cfg = {}
        self.method = ""
        self.rows = []
        self.saved = []

    def start_method(self, method):
        self.method, self.rows = method, []

    def grid(self, values):
        return list(values)

    def record(self, setting):
        nll = 1.75 + 0.1 * setting if self.method == "wiseft" else 1.9 - 0.01 * setting
        prompts = {"en": {"kl_sum": [0.1 * (i + 1) for i in range(20)], "n": [10] * 20}}
        self.rows.append({"setting": setting, "task_nll": nll, "S": {"en": 0.1}, "S_mean": 0.1, "S_prompts": prompts})

    def save(self, tag):
        self.saved.append(tag)


def test_one_failing_baseline_is_recorded_unavailable_and_the_pipeline_carries_on(tmp_path, monkeypatch, capsys):
    """RegMean raises in its dev grid and diagonal Fisher raises while matching: each is recorded as
    {"unavailable": ...} where it failed, and WiSE-FT is still measured, picked, matched and compared in P6."""
    baselines = _load_script("07_baselines")
    finalists = _load_script("09_pick_test_settings")
    p6 = _load_script("10_compare_with_baselines")
    results = tmp_path / "results"
    results.mkdir()

    failing = {"regmean"}

    def apply_setting(runner, method, value):
        if method in failing:
            raise RuntimeError("CUDA out of memory")

    ctx = type("Ctx", (), {"out": staticmethod(lambda *parts: tmp_path.joinpath(*parts)), "cfg": {}})()
    monkeypatch.setattr(baselines, "BaselineRunner", _StubRunner)
    monkeypatch.setattr(baselines, "setup", lambda args: ctx)
    monkeypatch.setattr(baselines, "measure_unrepaired", lambda runner: {"theta0": {"task_nll": 2.1}})
    monkeypatch.setattr(baselines, "apply_setting", apply_setting)
    for module in (finalists, p6):
        monkeypatch.setattr(module, "outputs_dir", lambda name: tmp_path / name)
    monkeypatch.setattr(baselines, "write_result", lambda name, result, **kw: _write(results, name, result))

    methods = ["--methods", "wiseft", "regmean", "diag_fisher"]
    baselines.main(["--run", "A", *methods, "--out-name", "dev"])
    dev = json.loads((results / "dev.json").read_text())["result"]
    assert dev["regmean"] == {"unavailable": "RuntimeError: CUDA out of memory", "rows": []}
    assert len(dev["wiseft"]) == len(baselines.WISEFT_ALPHAS) and len(dev["diag_fisher"]) == 11

    stub = _StubRunner(None, "A", [])
    stub.record(0.0)
    prompts = stub.rows[0]["S_prompts"]
    point = {"mu": 0.1, "task_nll": 1.80, "trust_region_failed": False, "S_actual": {"en": 0.05}, "S_prompts": prompts}
    sweep = {"task_nll": {"theta0": 2.1, "theta1": 1.75}, "points": [point]}
    (results / "sweep.json").write_text(json.dumps({"result": sweep}))
    finalists.main(["--run", "A", "--sweep", "sweep", "--baselines", "dev", "--out-name", "picks"])
    picks = json.loads((results / "picks.json").read_text())["result"]["baselines"]
    assert "unavailable" in picks["regmean"] and "setting" in picks["wiseft"] and "setting" in picks["diag_fisher"]

    failing.add("diag_fisher")  # measured on dev, then fails while being matched and saved
    baselines.main(["--run", "A", *methods, "--match-nll", "1.80", "--match-from", "dev", "--out-name", "matched"])
    matched = json.loads((results / "matched.json").read_text())["result"]["matched"]
    assert matched["regmean"] == {"unavailable": "dev grid failed: RuntimeError: CUDA out of memory"}
    assert matched["diag_fisher"] == {"unavailable": "RuntimeError: CUDA out of memory"}
    assert matched["wiseft"]["within_tolerance"]

    capsys.readouterr()
    p6.main(["--run", "A", "--sweep", "sweep", "--matched", "matched"])
    compared = json.loads((results / "p6_runA.json").read_text())["result"]["baselines"]
    assert compared["regmean"] == {"unavailable": "dev grid failed: RuntimeError: CUDA out of memory"}
    assert compared["diag_fisher"] == {"unavailable": "RuntimeError: CUDA out of memory"}
    assert "langs" in compared["wiseft"]


def _write(results, name, result):
    (results / f"{name}.json").write_text(json.dumps({"result": result}))
