"""scripts/10_p6.py on small result files: μ* point, paired per-language differences, Holm."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_p6():
    spec = importlib.util.spec_from_file_location("script_10_p6", SCRIPTS / "10_p6.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["script_10_p6"] = module
    spec.loader.exec_module(module)
    return module


def _prompts(per_token_kl, counts):
    return {"kl_sum": list(np.asarray(per_token_kl) * counts), "n": [int(n) for n in counts]}


def test_p6_compares_mu_star_with_each_matched_baseline(tmp_path, monkeypatch, capsys):
    p6 = _load_p6()
    r = np.random.default_rng(0)
    counts = r.integers(30, 120, 80)
    low = {lang: _prompts(r.exponential(0.05, 80), counts) for lang in ("en", "hi")}
    high = {lang: _prompts(np.asarray(low[lang]["kl_sum"]) / counts + 0.1, counts) for lang in ("en", "hi")}
    points = [  # μ = 0.01 fails rule 11; μ = 0.1 is the smallest accepted μ within the NLL limit 1.1
        {"mu": 0.01, "task_nll": 1.05, "trust_region_failed": True, "S_actual": {}, "S_prompts": high},
        {"mu": 0.1, "task_nll": 1.08, "trust_region_failed": False, "S_actual": {"en": 0.05}, "S_prompts": low},
        {"mu": 1.0, "task_nll": 1.02, "trust_region_failed": False, "S_actual": {}, "S_prompts": high},
    ]
    sweep = {"task_nll": {"theta0": 2.0, "theta1": 1.0}, "points": points}
    baseline = {"setting": 0.3, "task_nll": 1.081, "S": {"en": 0.15}, "S_prompts": high}
    matched = {
        "matched": {
            "wiseft": {"matched": baseline, "within_tolerance": True},
            "regmean": {"error": "OutOfMemoryError"},
        }
    }
    results = tmp_path / "results"
    results.mkdir()
    (results / "sweep.json").write_text(json.dumps({"result": sweep}))
    (results / "matched.json").write_text(json.dumps({"result": matched}))
    monkeypatch.setattr(p6, "outputs_dir", lambda name: tmp_path / name)

    p6.main(["--run", "A", "--sweep", "sweep", "--matched", "matched"])
    written = json.loads((results / "p6_runA.json").read_text())["result"]
    assert written["mu_star"] == 0.1
    for lang in ("en", "hi"):
        row = written["baselines"]["wiseft"]["langs"][lang]
        assert row["diff"] == pytest.approx(-0.1) and row["pencil_lower"] and not row["pencil_higher"]
        assert row["p_holm"] >= row["p_two_sided"]
    assert "unavailable" in written["baselines"]["regmean"]
    assert "pencil lower" in capsys.readouterr().out


def test_p6_refuses_unpaired_prompts():
    p6 = _load_p6()
    with pytest.raises(ValueError, match="same prompts"):
        p6.language_difference({"kl_sum": [1.0], "n": [3]}, {"kl_sum": [1.0], "n": [4]})


def test_p6_labels_a_backup_mu_star(tmp_path, monkeypatch, capsys):
    """No accepted μ within the NLL limit (1.1): P6 runs at the backup μ* and says so."""
    p6 = _load_p6()
    counts = np.full(10, 50)
    prompts = {"en": _prompts(np.full(10, 0.05), counts)}
    points = [
        {"mu": 0.1, "task_nll": 1.3, "trust_region_failed": False, "S_actual": {"en": 0.05}, "S_prompts": prompts},
        {"mu": 1.0, "task_nll": 1.2, "trust_region_failed": False, "S_actual": {"en": 0.05}, "S_prompts": prompts},
    ]
    baseline = {"setting": 0.3, "task_nll": 1.2, "S": {"en": 0.05}, "S_prompts": prompts}
    results = tmp_path / "results"
    results.mkdir()
    sweep = {"task_nll": {"theta0": 2.0, "theta1": 1.0}, "points": points}
    (results / "sweep.json").write_text(json.dumps({"result": sweep}))
    matched = {"matched": {"wiseft": {"matched": baseline, "within_tolerance": True}}}
    (results / "matched.json").write_text(json.dumps({"result": matched}))
    monkeypatch.setattr(p6, "outputs_dir", lambda name: tmp_path / name)
    p6.main(["--sweep", "sweep", "--matched", "matched"])
    written = json.loads((results / "p6_runA.json").read_text())["result"]
    assert written["mu_star"] == 1.0 and written["mu_star_backup"] is True
    assert "backup μ*" in capsys.readouterr().out
