"""Repair, μ sweep, trust-region check and forecast (Prompt 6; predictions P2 and P3).

sweep     For one run: repair every module at each μ, measure on dev the actual safety drift
          S = KL(θ0 ‖ θ) on θ0's own responses to dev harmful prompts and the task NLL on the task dev rows,
          and log predicted S/R, step ratios (Euclidean and input-weighted) and the task-gradient diagnostic.
          Each μ goes through the rule-11 trust region (covariant first, Euclidean fallback). --save writes
          merged checkpoints for chosen μ values to outputs/ckpt/repaired/ for 02_evaluate.py.
          --layers restricts the repair to some layers (the row-set comparison on 4 layers).
forecast  For every adapter on disk (runs × fractions): the forecast ½ Σ tr(G_h ΔW A_h ΔWᵀ) per language
          (harmful factors only), correlated with the measured refusal drop from 02_evaluate.py results
          named probe_theta0 and probe_run{R}_frac{F} (Spearman with a bootstrap CI). A ranking, not a number.

    python scripts/06_repair.py sweep --run A
    python scripts/06_repair.py sweep --run A --row-set all --layers selection      # row-set comparison
    python scripts/06_repair.py sweep --run A --save 0.1 0.3
    python scripts/06_repair.py forecast
    python scripts/06_repair.py sweep --run A --dry-run
"""
from __future__ import annotations

import argparse
import copy
import json

import torch

from pencil.delta import lora_delta, set_weights
from pencil.evaluate import task_nll
from pencil.factors import parse_name
from pencil.repair import ModelRepair, forecast_many, model_weights, task_grad_diagnostic, trust_region_repair
from pencil.results import write_result
from pencil.runtime import (available_langs, common_args, dev_harm_batches, encode_pairs, harm_factors, load_model,
                            load_tokenizer, make_batches, measure_S, prepare_factors, repair_kwargs, setup,
                            task_dev_rows, task_factors)

REPAIRED_LABEL = "research artifact: pencil-repaired checkpoint, mu={mu}, from run {run}"


def sweep(ctx, args):
    pc = ctx.cfg["pencil"]
    tok = load_tokenizer(ctx)
    model0 = load_model(ctx.aligned, ctx)
    adapter = ctx.out_root / "ckpt" / args.run / f"frac{args.frac}"
    deltas_all = lora_delta(adapter, lowrank=True)        # θ1 is always the full fine-tune
    selection = args.layers == "selection"
    langs = args.langs or available_langs(ctx, selection)
    fh, ft, tol, floors = prepare_factors(model0, harm_factors(ctx, langs, args.pool, selection),
                                          task_factors(ctx, args.run, args.frac, selection), pc)
    deltas = {n: d for n, d in deltas_all.items() if n in fh.a_key}
    modules = None
    if args.layers:
        keep = set(ctx.cfg["factors"]["selection_layers"]) if selection else {int(x) for x in args.layers.split(",")}
        modules = ModelRepair.order(n for n in deltas if parse_name(n)[1] in keep)
        deltas = {n: deltas[n] for n in modules}          # repair (and pay for bases) only there
    row_set = args.row_set or pc.get("repair_row_set", "all")
    rs_tag = row_set if isinstance(row_set, str) else "mixed"
    tag = f"run{args.run}_frac{args.frac}_{rs_tag}" + ("_" + args.layers.replace(",", "-") if args.layers else "")
    n_probe = ctx.cfg["evaluation"]["probe_per_lang"]
    dev_h = dev_harm_batches(ctx, model0, tok, langs, n_probe)
    task_dev = task_dev_rows(ctx)
    W0_all = model_weights(model0, deltas_all, device="cpu")             # CPU: keeps the GPU for two models
    theta1 = {n: (W0_all[n].double() + deltas_all[n].double()).to(W0_all[n].dtype) for n in deltas_all}   # bf16, CPU
    W0 = {n: W0_all[n] for n in deltas}

    work = copy.deepcopy(model0)
    set_weights(work, theta1)
    S_drift = measure_S(model0, work, dev_h)
    nll = {"theta0": task_nll(model0, tok, task_dev)["nll_per_token"], "theta1": task_nll(work, tok, task_dev)["nll_per_token"]}
    S_mean = lambda d: sum(d.values()) / len(d)
    print("drift S per lang " + ", ".join(f"{k} {v:.3g}" for k, v in S_drift.items())
          + f"; task NLL θ0 {nll['theta0']:.3f} θ1 {nll['theta1']:.3f}")

    def actual_S(new):
        if new is None:
            return S_mean(S_drift)
        set_weights(work, new)
        return S_mean(measure_S(model0, work, dev_h))

    common = repair_kwargs(pc, tol, row_set=row_set)
    common.pop("damping"), common.pop("gamma")
    repairers: dict = {}                                   # one per damping setting, reused across μ
    mus = args.mu or pc["mu_grid"]
    points = []
    for mu in mus:
        if modules is None:
            new, report, setting = trust_region_repair(fh, ft, W0, deltas, mu, gamma_c=pc["gamma_c"],
                                                       gamma_e_grid=pc["gamma_e_grid"], actual_S=actual_S,
                                                       repairers=repairers, **common)
        else:
            key = (pc["damping"], pc["gamma_c"])
            if key not in repairers:
                repairers[key] = ModelRepair(fh, ft, damping=key[0], gamma=key[1], **common)
            new, report = repairers[key].repair(W0, deltas, mu)
            setting = {"damping": key[0], "gamma": key[1], "attempts": []}
        set_weights(work, theta1)                          # modules outside `deltas` keep θ1's weights
        set_weights(work, new)
        S = measure_S(model0, work, dev_h)
        t = task_nll(work, tok, task_dev)["nll_per_token"]
        point = {"mu": mu, "damping": setting["damping"], "gamma": setting["gamma"], "trust_region": setting["attempts"],
                 "trust_region_failed": bool(setting.get("failed")),
                 "S_actual": S, "S_actual_mean": S_mean(S), "task_nll": t, "task_regret_measured": t - nll["theta1"],
                 "S_pred": report["S_after_pred"], "S_before_pred": report["S_before_pred"], "R_pred": report["R_pred"],
                 "max_step_ratio": report["max_step_ratio"], "max_input_step_ratio": report["max_input_step_ratio"],
                 "step_ratios": {n: m["step_ratio"] for n, m in report["modules"].items()},
                 "input_step_ratios": {n: m["input_step_ratio"] for n, m in report["modules"].items()}}
        if args.grad_diag:
            rep = repairers[(setting["damping"], setting["gamma"])]
            steps = rep.steps(deltas, mu, modules)          # same modules as R_pred
            set_weights(work, theta1)                       # the gradient is taken at θ1, on `work`
            enc = encode_pairs(tok, [(r["prompt"], r["answer"]) for r in task_dev[:64]], ctx.cfg["factors"]["max_len"])
            point["task_grad_diagnostic"] = task_grad_diagnostic(work, make_batches(enc, tok.pad_token_id, 1), steps,
                                                                 report["R_pred"])
            set_weights(work, new)
        points.append(point)
        print(f"mu={mu:g} [{setting['damping']} {setting['gamma']:g}]: S {S_mean(S):.4g} (drift {S_mean(S_drift):.4g}, "
              f"pred {report['S_after_pred']:.3g}), task NLL {t:.3f}, max step {report['max_step_ratio']:.2f} "
              f"(input-weighted {report['max_input_step_ratio']:.2f})"
              + ("  TRUST REGION FAILED at every damping: do not evaluate this point" if setting.get("failed") else ""))
        if args.save and any(abs(mu - x) < 1e-12 for x in args.save):
            d = ctx.out("ckpt", "repaired", f"{tag}_mu{mu:g}")
            work.save_pretrained(d)
            tok.save_pretrained(d)
            (d / "RESEARCH_ARTIFACT.txt").write_text(REPAIRED_LABEL.format(mu=mu, run=args.run) + "\n")
            print(f"  saved {d}")

    key = (pc["damping"], pc["gamma_c"])
    if key not in repairers:
        repairers[key] = ModelRepair(fh, ft, damping=key[0], gamma=key[1], **common)
    frontier = repairers[key].frontier(deltas, mus)
    try:
        from pencil.stats import spearman_bootstrap_ci
        rho = spearman_bootstrap_ci([p["S_pred"] for p in points], [p["S_actual_mean"] for p in points])
    except Exception as e:                                     # too few points for a CI
        rho = {"error": str(e)}
    name = f"repair_sweep_{tag}"
    path = write_result(name, {"points": points, "S_drift": S_drift, "task_nll": nll, "frontier_pred": frontier,
                               "spearman_pred_vs_actual_S": rho, "rel_tol": tol, "null_floors": floors,
                               "langs": langs, "row_set": row_set, "modules": modules, "frac": args.frac},
                        config={"pencil": pc}, checkpoint=str(adapter), out_dir=ctx.out("results"))
    plot_sweep(ctx, points, S_drift, nll, name)
    print(f"-> {path}")


def plot_sweep(ctx, points, S_drift, nll, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    for l in S_drift:
        ax[0].plot([p["task_nll"] for p in points], [p["S_actual"][l] for p in points], "o-", label=l)
    ax[0].axhline(0, color="grey", lw=0.5)
    ax[0].set_xlabel("task NLL on dev (lower is better)")
    ax[0].set_ylabel("measured S = KL(θ0‖θ) on dev")
    ax[0].set_title("measured frontier", fontsize=9)
    ax[0].legend(fontsize=7)
    ax[1].loglog([max(p["S_pred"], 1e-12) for p in points], [max(p["S_actual_mean"], 1e-12) for p in points], "o")
    ax[1].set_xlabel("predicted S (K-FAC, a ranking)")
    ax[1].set_ylabel("measured S")
    ax[1].set_title("prediction vs measurement", fontsize=9)
    fig.tight_layout()
    fig.savefig(ctx.out("figures") / f"{name}.png", dpi=160)
    plt.close(fig)


def forecast(ctx, args):
    """One streaming pass over each language's harmful factors, forecasting every checkpoint at once."""
    pc = ctx.cfg["pencil"]
    model0 = load_model(ctx.aligned, ctx)
    langs = args.langs or available_langs(ctx)
    row_set = pc.get("forecast_row_set", "all")
    rs_of = (lambda n: row_set) if isinstance(row_set, str) else \
        (lambda n: row_set.get(n.rsplit(".", 1)[-1], row_set.get("default", "all")))
    ckpts = {}
    for run in ctx.cfg["runs"]:
        for frac in ctx.cfg["save_fractions"]:
            ad = ctx.out_root / "ckpt" / run / f"frac{round(frac * 100)}"
            if ad.exists():
                ckpts[(run, frac)] = lora_delta(ad, lowrank=True)
            else:
                print(f"missing adapter {ad}: run 03_finetune.py --run {run} first")
    res_dir = ctx.out_root / "results"
    base = _eval(res_dir, "probe_theta0")
    rows = []
    for l in langs:
        fh, _, _, _ = prepare_factors(model0, harm_factors(ctx, [l]), harm_factors(ctx, [l]), pc)
        fc = forecast_many(fh, {f"{r}|{f}": d for (r, f), d in ckpts.items()}, rs_of)
        for (run, frac) in ckpts:
            ev = _eval(res_dir, f"probe_run{run}_frac{round(frac * 100)}")
            drop = base[l] - ev[l] if base and ev and l in base and l in ev else None
            total = sum(fc[f"{run}|{frac}"].values())
            rows.append({"run": run, "frac": frac, "lang": l, "forecast": total, "refusal_drop": drop,
                         "per_module": fc[f"{run}|{frac}"]})
            print(f"{run} {frac:.2f} {l}: forecast {total:.4g}" + (f", refusal drop {drop:+.3f}" if drop is not None else ""))
    pairs = [(r["forecast"], r["refusal_drop"]) for r in rows if r["refusal_drop"] is not None]
    rho = None
    if len(pairs) >= 4:
        from pencil.stats import spearman_bootstrap_ci
        rho = spearman_bootstrap_ci([a for a, _ in pairs], [b for _, b in pairs])
        print(f"Spearman(forecast, refusal drop) over {len(pairs)} checkpoint-language pairs: {rho}")
    write_result("forecast_p2", {"rows": rows, "spearman": rho, "row_set": row_set, "langs": langs},
                 config={"pencil": pc}, out_dir=ctx.out("results"))


def _eval(res_dir, name):
    """{lang: refusal rate} from a 02_evaluate.py result, or None."""
    p = res_dir / f"{name}.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())["result"]["metrics"]
    return {l: v["harmful"]["refusal"]["p"] for l, v in m.items() if "harmful" in v}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = common_args(sub.add_parser("sweep"))
    s.add_argument("--run", default="A")
    s.add_argument("--frac", type=int, default=100)
    s.add_argument("--langs", nargs="+")
    s.add_argument("--pool", choices=["equal", "tokens"], default="equal")
    s.add_argument("--row-set", default=None)
    s.add_argument("--layers", default=None, help="'selection' or e.g. '3,11': repair only these layers")
    s.add_argument("--mu", type=float, nargs="+", help="default pencil.mu_grid")
    s.add_argument("--save", type=float, nargs="+", help="save merged checkpoints for these μ")
    s.add_argument("--grad-diag", action="store_true", help="task-gradient diagnostic (needs weight grads)")
    f = common_args(sub.add_parser("forecast"))
    f.add_argument("--langs", nargs="+")
    args = p.parse_args(argv)
    ctx = setup(args)
    (sweep if args.cmd == "sweep" else forecast)(ctx, args)


if __name__ == "__main__":
    main()
