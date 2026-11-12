"""The finalist rule (pencil.finalists), 09_finalists.py on small result files, and 07_baselines.py's --finalists
grid restriction."""

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


def test_mu_star_skips_failed_trust_regions_and_errors_and_refuses_when_nothing_qualifies():
    points = _points({0.1: 1.05, 0.3: 1.03}, failed={0.1}) + [{"error": "RuntimeError: ..."}]
    assert pencil_mu_star(points, NLL_THETA0, NLL_THETA1)["mu"] == 0.3
    with pytest.raises(ValueError, match="keeps 90%"):
        pencil_mu_star(_points({0.1: 1.5}), NLL_THETA0, NLL_THETA1)


def test_baseline_finalist_is_the_setting_closest_in_task_nll():
    rows = [{"setting": 0.3, "task_nll": 1.4}, {"error": "boom"}, {"setting": 3.0, "task_nll": 1.09}]
    assert closest_setting(rows, target_nll=1.08)["setting"] == 3.0


def test_finalists_script_writes_the_picks_and_prints_them_for_the_shell(tmp_path, monkeypatch, capsys):
    finalists = _load_script("09_finalists")
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
    finalists = _load_script("09_finalists")
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
    finalists = _load_script("09_finalists")
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
