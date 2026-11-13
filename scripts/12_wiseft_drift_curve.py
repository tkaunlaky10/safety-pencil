#!/usr/bin/env python
"""How drift shrinks along WiSE-FT (13 Nov, for the draft; CPU only, reads result JSONs).

Along W0 + (1 − α)ΔW the quadratic model predicts S(α) = S(θ1)·(1 − α)². We fit p in S(α)/S(θ1) = (1 − α)^p
(least squares through the origin on log–log axes, since a pure power law is exactly 1 at α = 0), per language and
for the mean over languages, with a 95% bootstrap interval over prompts: each replicate resamples each language's
prompts once, shared by θ1 and every α, and recomputes the token-weighted S = Σ kl_sum / Σ n.

Also lists the task NLL along α next to the pencil sweep's task NLL and |gᵀδ| / (½ δᵀĤδ) per μ.

    python scripts/12_wiseft_drift_curve.py --run A --baselines baselines_runA_scaled --sweep repair_sweep_scaled_runA
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from pencil.config import outputs_dir
from pencil.finalists import baseline_rows
from pencil.plots import save, subplots
from pencil.results import write_result

N_BOOT = 2000
ALPHA = 0.05


def load_result(name: str) -> dict:
    """The payload of outputs/results/<name>.json."""
    return json.loads((outputs_dir("results") / f"{name}.json").read_text())["result"]


def fit_exponent(one_minus_alpha: np.ndarray, ratio: np.ndarray) -> float:
    """p minimising Σ (log ratio − p log(1 − α))²: least squares through the origin on log–log axes."""
    log_x = np.log(one_minus_alpha)
    log_y = np.log(ratio)
    return float((log_x * log_y).sum() / (log_x**2).sum())


def token_mean(prompts: dict, index: np.ndarray) -> float:
    """Token-weighted S over the prompts at `index` (with repeats)."""
    sums = np.asarray(prompts["kl_sum"], dtype=float)[index]
    counts = np.asarray(prompts["n"], dtype=float)[index]
    return float(sums.sum() / counts.sum())


def ratios(theta1: dict, rows: list[dict], langs: list[str], indices: dict) -> dict[str, np.ndarray]:
    """{lang or "mean": S(α)/S(θ1) for each row}, using each language's resampled prompt `indices`."""
    out = {}
    S_theta1 = {lang: token_mean(theta1[lang], indices[lang]) for lang in langs}
    S_rows = {lang: np.array([token_mean(row["S_prompts"][lang], indices[lang]) for row in rows]) for lang in langs}
    for lang in langs:
        out[lang] = S_rows[lang] / S_theta1[lang]
    mean_theta1 = np.mean([S_theta1[lang] for lang in langs])
    out["mean"] = np.mean([S_rows[lang] for lang in langs], axis=0) / mean_theta1
    return out


def exponent_with_ci(theta1: dict, rows: list[dict], seed: int = 0) -> dict:
    """{lang or "mean": {"p", "lo", "hi", "ratio": [...]}} with the bootstrap over prompts described above."""
    langs = list(theta1)
    one_minus_alpha = np.array([1.0 - row["setting"] for row in rows])
    full = {lang: np.arange(len(theta1[lang]["n"])) for lang in langs}
    point = ratios(theta1, rows, langs, full)
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in point}
    for _ in range(N_BOOT):
        indices = {lang: rng.integers(0, len(full[lang]), len(full[lang])) for lang in langs}
        for key, ratio in ratios(theta1, rows, langs, indices).items():
            samples[key].append(fit_exponent(one_minus_alpha, ratio))
    result = {}
    for key, ratio in point.items():
        lo, hi = np.percentile(samples[key], [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
        result[key] = {
            "p": fit_exponent(one_minus_alpha, ratio),
            "lo": float(lo),
            "hi": float(hi),
            "ratio": list(ratio),
        }
    return result


def wiseft_rows(dev_baselines: dict) -> list[dict]:
    """WiSE-FT's measured dev rows with 0 < α < 1, in increasing α."""
    rows = [row for row in baseline_rows(dev_baselines.get("wiseft", [])) if "S_prompts" in row]
    return sorted((row for row in rows if 0.0 < row["setting"] < 1.0), key=lambda row: row["setting"])


def nll_side_by_side(rows: list[dict], sweep: dict) -> dict:
    """WiSE-FT's task NLL per α and the pencil's task NLL and |gᵀδ| ratio per μ."""
    pencil = []
    for point in sweep["points"]:
        diagnostic = point.get("task_grad_diagnostic", {})
        pencil.append({"mu": point["mu"], "task_nll": point["task_nll"], "grad_ratio": diagnostic.get("ratio")})
    return {"wiseft": [{"alpha": row["setting"], "task_nll": row["task_nll"]} for row in rows], "pencil": pencil}


def plot(path, rows: list[dict], fits: dict) -> None:
    """S(α)/S(θ1) against 1 − α per language, with the (1 − α) and (1 − α)² references."""
    fig, ax = subplots(figsize=(4.5, 3.4))
    x = np.array([1.0 - row["setting"] for row in rows])
    grid = np.linspace(0.0, 1.0, 101)
    ax.plot(grid, grid, "--", color="0.6", label="(1 − α)")
    ax.plot(grid, grid**2, ":", color="0.4", label="(1 − α)², quadratic model")
    for key, fit in fits.items():
        ax.plot(x, fit["ratio"], "o-", label=f"{key}: p = {fit['p']:.2f} [{fit['lo']:.2f}, {fit['hi']:.2f}]")
    ax.set(xlabel="1 − α (share of ΔW kept)", ylabel="S(α) / S(θ1) on dev")
    ax.legend(fontsize=6)
    save(fig, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="A")
    parser.add_argument("--baselines", required=True, help="the dev baselines' result name")
    parser.add_argument("--sweep", required=True, help="the scaled sweep's result name")
    args = parser.parse_args(argv)
    dev = load_result(args.baselines)
    rows = wiseft_rows(dev)
    if len(rows) < 2:
        raise SystemExit("need at least two measured WiSE-FT settings with 0 < α < 1")
    fits = exponent_with_ci(dev["drift"]["S_prompts"], rows)
    result = {
        "alpha": [row["setting"] for row in rows],
        "fits": fits,
        "task_nll": nll_side_by_side(rows, load_result(args.sweep)),
        "n_boot": N_BOOT,
    }
    for key, fit in fits.items():
        print(f"{key}: p = {fit['p']:.3f} [{fit['lo']:.3f}, {fit['hi']:.3f}]")
    name = f"wiseft_shape_run{args.run}"
    write_result(
        name, result, config={"baselines": args.baselines, "sweep": args.sweep}, out_dir=outputs_dir("results")
    )
    plot(outputs_dir("figures") / f"{name}.png", rows, fits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
