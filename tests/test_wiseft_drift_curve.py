"""scripts/12_wiseft_drift_curve.py: the exponent fit and its bootstrap, on synthetic per-prompt drift."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load():
    spec = importlib.util.spec_from_file_location("script_12_wiseft_drift_curve", SCRIPTS / "12_wiseft_drift_curve.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prompts(per_token, counts):
    return {"kl_sum": list(np.asarray(per_token) * counts), "n": [int(n) for n in counts]}


def test_fit_recovers_a_known_exponent():
    shape = _load()
    x = np.array([0.95, 0.9, 0.8, 0.5, 0.25])
    assert shape.fit_exponent(x, x**1.5) == pytest.approx(1.5)


def test_script_fits_p_per_language_with_a_bootstrap_interval(tmp_path, monkeypatch):
    """Every prompt's drift shrinks as (1 − α)^1.2 times prompt-level noise: p ≈ 1.2, inside its interval."""
    shape = _load()
    r = np.random.default_rng(0)
    counts = r.integers(20, 100, 60)
    base = {lang: r.exponential(0.25, 60) for lang in ("en", "hi")}
    theta1 = {lang: _prompts(base[lang], counts) for lang in base}
    rows = []
    for alpha in (0.1, 0.3, 0.5, 0.75):
        prompts = {lang: _prompts(base[lang] * (1 - alpha) ** 1.2 * r.lognormal(0, 0.05, 60), counts) for lang in base}
        rows.append({"setting": alpha, "task_nll": 1.78, "S_mean": 0.1, "S_prompts": prompts})
    dev = {"drift": {"S_prompts": theta1}, "wiseft": rows}
    sweep = {"points": [{"mu": 0.0019, "task_nll": 1.78, "task_grad_diagnostic": {"ratio": 4.7}}]}
    results = tmp_path / "results"
    results.mkdir()
    (results / "dev.json").write_text(json.dumps({"result": dev}))
    (results / "sweep.json").write_text(json.dumps({"result": sweep}))
    monkeypatch.setattr(shape, "outputs_dir", lambda name: (tmp_path / name).mkdir(exist_ok=True) or tmp_path / name)
    monkeypatch.setattr(shape, "N_BOOT", 200)
    shape.main(["--baselines", "dev", "--sweep", "sweep"])
    written = json.loads((results / "wiseft_shape_runA.json").read_text())["result"]
    for key in ("en", "hi", "mean"):
        fit = written["fits"][key]
        assert fit["lo"] < fit["p"] < fit["hi"] and abs(fit["p"] - 1.2) < 0.05
    assert written["task_nll"]["pencil"] == [{"mu": 0.0019, "task_nll": 1.78, "grad_ratio": 4.7}]
    assert (tmp_path / "figures" / "wiseft_shape_runA.png").exists()
