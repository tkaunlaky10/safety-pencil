#!/usr/bin/env python
"""Pick the finalists of the test-tier comparison by the fixed rule in pencil.finalists (dev results only).

    python scripts/09_pick_test_settings.py --run A --sweep repair_sweep_compare_runA --baselines baselines_runA_must
    python scripts/09_pick_test_settings.py ... --print mu           # μ*, as in 06_repair.py's checkpoint names
    python scripts/09_pick_test_settings.py ... --print baselines    # method=value pairs (07_baselines.py --finalists)
    python scripts/09_pick_test_settings.py ... --print target_nll   # NLL(μ*), for 07_baselines.py --match-nll
    python scripts/09_pick_test_settings.py --sweep repair_sweep_compare_runA --any-passing  # 0 if a μ passed rule 11

Writes outputs/results/finalists_run<run>.json: the rule's numbers, μ*, and every compared baseline's pick.
Only the methods named by --test-tier are printed for evaluation; the others are still picked and recorded.
"""

from __future__ import annotations

import argparse
import json
import sys

from pencil.config import outputs_dir
from pencil.finalists import KEEP_TASK_GAIN, accepted, baseline_rows, closest_setting, pencil_mu_star
from pencil.results import write_result

COMPARED = ("regmean", "diag_fisher", "wiseft")


def load_result(name: str) -> dict:
    """The payload of outputs/results/<name>.json."""
    return json.loads((outputs_dir("results") / f"{name}.json").read_text())["result"]


def pick_baseline(rows: list[dict], target_nll: float) -> dict:
    """One baseline's pick, or {"unavailable": reason} if it has no measured setting (e.g. it failed on dev), so
    one broken baseline doesn't stop the others' comparison."""
    try:
        row = closest_setting(rows, target_nll)
    except ValueError as error:
        return {"unavailable": str(error)}
    return {"setting": row["setting"], "task_nll": row["task_nll"], "S_mean": row["S_mean"]}


def pick(sweep: dict, baselines: dict) -> dict:
    """μ* from the sweep and, for each compared baseline, its setting closest to NLL(μ*) in dev task NLL."""
    nll_theta0 = sweep["task_nll"]["theta0"]
    nll_theta1 = sweep["task_nll"]["theta1"]
    mu_star = pencil_mu_star(sweep["points"], nll_theta0, nll_theta1)
    picks = {}
    for method in COMPARED:
        picks[method] = pick_baseline(baseline_rows(baselines.get(method, [])), mu_star["task_nll"])
    return {
        "keep_task_gain": KEEP_TASK_GAIN,
        "task_nll": {"theta0": nll_theta0, "theta1": nll_theta1},
        "mu_star": mu_star,
        "baselines": picks,
    }


def any_mu_passed(sweep_name: str) -> bool:
    """True if the sweep's result exists and at least one μ passed rule 11 (False if the sweep wrote nothing)."""
    path = outputs_dir("results") / f"{sweep_name}.json"
    if not path.exists():
        return False
    points = json.loads(path.read_text())["result"]["points"]
    return any(accepted(point) for point in points)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="the run (needed unless --any-passing)")
    parser.add_argument("--sweep", required=True, help="the sweep's result name")
    parser.add_argument("--baselines", help="the must-run baselines' result name (needed unless --any-passing)")
    parser.add_argument("--any-passing", action="store_true", help="only check: exit 0 if any μ passed rule 11")
    parser.add_argument("--test-tier", nargs="+", default=list(COMPARED), choices=COMPARED)
    parser.add_argument(
        "--print", dest="show", choices=["mu", "baselines", "target_nll"], help="print one pick for the shell"
    )
    parser.add_argument("--out-name", default=None, help="result name (default finalists_run<run>)")
    args = parser.parse_args(argv)
    if args.any_passing:
        return 0 if any_mu_passed(args.sweep) else 1
    if not (args.run and args.baselines):
        parser.error("--run and --baselines are needed to pick the finalists")

    finalists = pick(load_result(args.sweep), load_result(args.baselines))
    finalists["test_tier"] = args.test_tier
    if args.show == "mu":
        print(f"{finalists['mu_star']['mu']:g}")  # as 06_repair.py names its checkpoint folders
        return 0
    if args.show == "target_nll":
        print(repr(finalists["mu_star"]["task_nll"]))
        return 0
    if args.show == "baselines":
        available = [method for method in args.test_tier if "setting" in finalists["baselines"][method]]
        pairs = [f"{method}={finalists['baselines'][method]['setting']!r}" for method in available]
        print(" ".join(pairs))
        return 0
    config = {"sweep": args.sweep, "baselines": args.baselines}
    write_result(args.out_name or f"finalists_run{args.run}", finalists, config=config, out_dir=outputs_dir("results"))
    print(json.dumps(finalists, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
