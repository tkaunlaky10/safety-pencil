#!/usr/bin/env python
"""Does the pencil drift less than the baselines at the same task score? (prediction P6 in PREREGISTRATION.md)

P6 on dev (13 Nov): the scaled pencil at μ* against each baseline at matched dev task NLL. Output files keep their
original names (p6_run<run>, p6_frontier_run<run>) so earlier results still load.

Per language, the difference in measured drift S (pencil − baseline; negative favours the pencil) with a paired
bootstrap over prompts: both models are scored on the same θ0 answers, so each prompt's summed KL pairs up and
S = Σ kl_sum / Σ n (stats.paired_bootstrap_token_mean_diff). 95% intervals; Holm across the languages within
each baseline. μ* is the fixed rule (pencil.finalists); the baselines' matched rows come from 07_baselines.py
--match-nll. Reads result JSONs only (no GPU).

--frontier DEV_BASELINES (post-hoc, 13 Nov; PREREGISTRATION.md deviation log): instead of the matched rows, each
baseline's lowest-S dev setting with task NLL ≤ NLL(μ*) + MATCH_TOLERANCE (pencil.finalists.frontier_setting),
written to p6_frontier_run<run>. This is the headline P6; the pre-registered closest-NLL P6 is reported beside it.

    python scripts/10_compare_with_baselines.py --sweep repair_sweep_scaled_runA --matched baselines_runA_matched
    python scripts/10_compare_with_baselines.py --sweep repair_sweep_scaled_runA --frontier baselines_runA_scaled
"""

from __future__ import annotations

import argparse
import json
import sys

from pencil.config import outputs_dir
from pencil.finalists import MATCH_TOLERANCE, baseline_rows, frontier_setting, pencil_mu_star
from pencil.results import write_result
from pencil.stats import holm, paired_bootstrap_token_mean_diff

ALPHA = 0.05


def load_result(name: str) -> dict:
    """The payload of outputs/results/<name>.json."""
    return json.loads((outputs_dir("results") / f"{name}.json").read_text())["result"]


def mu_star_point(sweep: dict) -> tuple[dict, bool]:
    """(the sweep point at μ*, whether it is the backup μ*) by the rule in pencil.finalists."""
    mu_star = pencil_mu_star(sweep["points"], sweep["task_nll"]["theta0"], sweep["task_nll"]["theta1"])
    point = next(point for point in sweep["points"] if point["mu"] == mu_star["mu"])
    return point, mu_star["backup"]


def language_difference(pencil_prompts: dict, baseline_prompts: dict) -> dict:
    """The paired bootstrap of S(pencil) − S(baseline) for one language's per-prompt KL sums."""
    if pencil_prompts["n"] != baseline_prompts["n"]:
        raise ValueError("pencil and baseline were not scored on the same prompts and tokens")
    return paired_bootstrap_token_mean_diff(pencil_prompts["kl_sum"], baseline_prompts["kl_sum"], pencil_prompts["n"])


def compare(pencil_point: dict, baseline_row: dict) -> dict:
    """{lang: {"diff", "lo", "hi", "p_two_sided", "p_holm", "pencil_lower", "pencil_higher"}} plus the NLL gap."""
    langs = list(pencil_point["S_prompts"])
    per_lang = {}
    for lang in langs:
        per_lang[lang] = language_difference(pencil_point["S_prompts"][lang], baseline_row["S_prompts"][lang])
    adjusted = holm([per_lang[lang]["p_two_sided"] for lang in langs])
    for lang, p_holm in zip(langs, adjusted):
        row = per_lang[lang]
        row["p_holm"] = p_holm
        row["pencil_lower"] = p_holm < ALPHA and row["hi"] < 0
        row["pencil_higher"] = p_holm < ALPHA and row["lo"] > 0
    return {
        "langs": per_lang,
        "task_nll_pencil": pencil_point["task_nll"],
        "task_nll_baseline": baseline_row["task_nll"],
        "setting": baseline_row["setting"],
        "S_pencil": pencil_point["S_actual"],
        "S_baseline": baseline_row["S"],
    }


def p6(sweep: dict, matched: dict) -> dict:
    """The comparison against every matched baseline; one without a matched row is recorded as unavailable."""
    point, backup = mu_star_point(sweep)
    comparisons = {}
    for method, result in matched["matched"].items():
        if "matched" not in result:
            comparisons[method] = {"unavailable": result.get("unavailable", "no matched row")}
            continue
        comparisons[method] = compare(point, result["matched"])
        comparisons[method]["within_tolerance"] = result["within_tolerance"]
    return {"mu_star": point["mu"], "mu_star_backup": backup, "alpha": ALPHA, "baselines": comparisons}


def p6_frontier(sweep: dict, dev_baselines: dict, methods=("wiseft", "regmean", "diag_fisher")) -> dict:
    """The post-hoc frontier comparison: μ* against each baseline's lowest-S dev setting that is at least as good
    on the task (NLL ≤ NLL(μ*) + MATCH_TOLERANCE). A baseline with no such setting is recorded as unavailable."""
    point, backup = mu_star_point(sweep)
    comparisons = {}
    for method in methods:
        entry = dev_baselines.get(method, [])
        row = frontier_setting(baseline_rows(entry), point["task_nll"])
        if row is None:
            reason = entry.get("unavailable") if isinstance(entry, dict) else None
            comparisons[method] = {"unavailable": reason or "no dev setting with task NLL ≤ NLL(μ*) + tolerance"}
            continue
        comparisons[method] = compare(point, row)
    return {
        "mu_star": point["mu"],
        "mu_star_backup": backup,
        "alpha": ALPHA,
        "rule": "post-hoc frontier: lowest-S dev setting with task NLL <= NLL(mu*) + tolerance",
        "tolerance": MATCH_TOLERANCE,
        "baselines": comparisons,
    }


def print_table(result: dict) -> None:
    kind = "post-hoc frontier" if "rule" in result else "pre-registered closest NLL"
    where = f"{mu_star_label(result)} = {result['mu_star']:g}"
    print(f"P6 ({kind}) on dev at {where}: S(pencil) − S(baseline), 95% CI, Holm p")
    for method, comparison in result["baselines"].items():
        if "unavailable" in comparison:
            print(f"  {method}: unavailable ({comparison['unavailable']})")
            continue
        gap = comparison["task_nll_baseline"] - comparison["task_nll_pencil"]
        print(f"  {method} (setting {comparison['setting']:g}, NLL gap {gap:+.4f}):")
        for lang, row in comparison["langs"].items():
            interval = f"{row['diff']:+.4g} [{row['lo']:+.4g}, {row['hi']:+.4g}]"
            print(f"    {lang}: {interval} p_holm {row['p_holm']:.3g} {verdict_of(row)}")


def mu_star_label(result: dict) -> str:
    """'backup μ*' when no accepted μ kept 90% of the task gain, else 'μ*'."""
    if result.get("mu_star_backup"):
        return "backup μ*"
    return "μ*"


def verdict_of(row: dict) -> str:
    """'pencil lower', 'pencil higher' (Holm p < ALPHA and the interval excludes 0) or 'n.s.'."""
    if row["pencil_lower"]:
        return "pencil lower"
    if row["pencil_higher"]:
        return "pencil higher"
    return "n.s."


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="A")
    parser.add_argument("--sweep", required=True, help="the scaled sweep's result name")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--matched", help="07_baselines.py --match-nll's result name (the pre-registered P6)")
    which.add_argument("--frontier", help="the dev baselines' result name (the post-hoc frontier P6)")
    args = parser.parse_args(argv)
    sweep = load_result(args.sweep)
    if args.frontier:
        result = p6_frontier(sweep, load_result(args.frontier))
        name, config = f"p6_frontier_run{args.run}", {"sweep": args.sweep, "frontier": args.frontier}
    else:
        result = p6(sweep, load_result(args.matched))
        name, config = f"p6_run{args.run}", {"sweep": args.sweep, "matched": args.matched}
    print_table(result)
    write_result(name, result, config=config, out_dir=outputs_dir("results"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
