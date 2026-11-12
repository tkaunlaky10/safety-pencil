#!/usr/bin/env python
"""Pick the finalists of the test-tier comparison by the fixed rule in pencil.finalists (dev results only).

    python scripts/09_finalists.py --run A --sweep repair_sweep_compare_runA --baselines baselines_runA_must
    python scripts/09_finalists.py ... --print mu           # μ*, for 06_repair.py sweep --save
    python scripts/09_finalists.py ... --print baselines    # method=value pairs, for 07_baselines.py --finalists

Writes outputs/results/finalists_run<run>.json: the rule's numbers, μ*, and every compared baseline's pick.
Only the methods named by --test-tier are printed for evaluation; the others are still picked and recorded.
"""

from __future__ import annotations

import argparse
import json
import sys

from pencil.config import outputs_dir
from pencil.finalists import KEEP_TASK_GAIN, closest_setting, pencil_mu_star
from pencil.results import write_result

COMPARED = ("regmean", "diag_fisher", "wiseft")


def load_result(name: str) -> dict:
    """The payload of outputs/results/<name>.json."""
    return json.loads((outputs_dir("results") / f"{name}.json").read_text())["result"]


def pick(sweep: dict, baselines: dict) -> dict:
    """μ* from the sweep and, for each compared baseline, its setting closest to NLL(μ*) in dev task NLL."""
    nll_theta0 = sweep["task_nll"]["theta0"]
    nll_theta1 = sweep["task_nll"]["theta1"]
    mu_star = pencil_mu_star(sweep["points"], nll_theta0, nll_theta1)
    picks = {}
    for method in COMPARED:
        row = closest_setting(baselines[method], mu_star["task_nll"])
        picks[method] = {"setting": row["setting"], "task_nll": row["task_nll"], "S_mean": row["S_mean"]}
    return {
        "keep_task_gain": KEEP_TASK_GAIN,
        "task_nll": {"theta0": nll_theta0, "theta1": nll_theta1},
        "mu_star": mu_star,
        "baselines": picks,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--sweep", required=True, help="the sweep's result name")
    parser.add_argument("--baselines", required=True, help="the must-run baselines' result name")
    parser.add_argument("--test-tier", nargs="+", default=list(COMPARED), choices=COMPARED)
    parser.add_argument("--print", dest="show", choices=["mu", "baselines"], help="print one pick for the shell")
    args = parser.parse_args(argv)

    finalists = pick(load_result(args.sweep), load_result(args.baselines))
    finalists["test_tier"] = args.test_tier
    if args.show == "mu":
        print(repr(finalists["mu_star"]["mu"]))
        return 0
    if args.show == "baselines":
        pairs = [f"{method}={finalists['baselines'][method]['setting']!r}" for method in args.test_tier]
        print(" ".join(pairs))
        return 0
    config = {"sweep": args.sweep, "baselines": args.baselines}
    write_result(f"finalists_run{args.run}", finalists, config=config, out_dir=outputs_dir("results"))
    print(json.dumps(finalists, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
