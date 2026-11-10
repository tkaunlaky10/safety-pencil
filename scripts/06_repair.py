"""Repair, μ sweep, trust-region check and forecast (Prompt 6; predictions P2 and P3).

sweep     For one run, repair every module at each μ through the rule-11 trust region (covariant first,
          Euclidean fallback), and measure on dev the safety drift S = KL(θ0 ‖ θ) on θ0's own answers to
          harmful prompts and the task NLL. Logs predicted S and R, step ratios (Euclidean and
          input-weighted) and, with --grad-diag, the task-gradient diagnostic. --save writes merged
          checkpoints for chosen μ to outputs/ckpt/repaired/ for 02_evaluate.py. --layers repairs only some
          layers (the row-set comparison on the selection layers).
forecast  For every adapter on disk (runs × fractions) and language: ½ Σ tr(G_h ΔW A_h ΔWᵀ), correlated with
          the measured refusal drop from 02_evaluate.py results named probe_theta0 and probe_run{R}_frac{F}.
          A ranking, not a number.

    python scripts/06_repair.py sweep --run A
    python scripts/06_repair.py sweep --run A --layers selection --row-set all     # row-set comparison
    python scripts/06_repair.py sweep --run A --save 0.1 0.3
    python scripts/06_repair.py forecast
    python scripts/06_repair.py sweep --run A --dry-run
"""

from __future__ import annotations

import argparse
import copy
import json

import torch

from pencil.delta import as_dense, lora_delta, model_weights, save_checkpoint, set_weights
from pencil.modules import layer_order, parse_name
from pencil.plots import save, subplots
from pencil.repair import ModelRepair, forecast_many, task_grad_diagnostic, trust_region_repair
from pencil.results import write_result
from pencil.runtime import (
    DevProbe,
    adapter_dir,
    available_langs,
    common_args,
    encode_pairs,
    harm_factors,
    load_model,
    load_repair_inputs,
    make_batches,
    prepare_factors,
    repair_kwargs,
    setup,
)

REPAIRED_LABEL = "research artifact: pencil-repaired checkpoint, mu={mu}, from run {run}"


def sweep(ctx, args):
    pc = ctx.cfg["pencil"]
    selection = args.layers == "selection"
    inputs = load_repair_inputs(ctx, args.run, args.frac, langs=args.langs, pool=args.pool, selection=selection)
    model0, tok, deltas = inputs.model0, inputs.tok, inputs.deltas
    modules = None
    if args.layers:  # repair (and solve bases for) these layers only
        keep = set(ctx.cfg["factors"]["selection_layers"]) if selection else {int(x) for x in args.layers.split(",")}
        modules = [n for n in deltas if parse_name(n)[1] in keep]
        deltas = {n: deltas[n] for n in modules}
    row_set = args.row_set or pc.get("repair_row_set", "all")
    tag = f"run{args.run}_frac{args.frac}_{row_set if isinstance(row_set, str) else 'mixed'}" + (
        f"_{args.layers.replace(',', '-')}" if args.layers else ""
    )

    # θ1 is always the whole fine-tune, even when only some layers are repaired; weights stay on the CPU
    full_delta = lora_delta(adapter_dir(ctx, args.run, args.frac), lowrank=True)
    W0_all = model_weights(model0, full_delta, device="cpu")
    theta1 = {n: (W0_all[n].double() + as_dense(d, torch.float64)).to(W0_all[n].dtype) for n, d in full_delta.items()}
    W0 = {n: W0_all[n] for n in deltas}
    work = set_weights(copy.deepcopy(model0), theta1)
    probe = DevProbe(ctx, model0, tok, inputs.langs)
    drift = probe.measure(work)
    nll_theta0 = probe.task_nll(model0)
    print(
        "drift S "
        + ", ".join(f"{k} {v:.3g}" for k, v in drift["S"].items())
        + f"; task NLL θ0 {nll_theta0:.3f} θ1 {drift['task_nll']:.3f}"
    )

    def measured_S(new):  # for the trust region; None means unrepaired
        if new is None:
            return drift["S_mean"]
        set_weights(work, new)
        return probe.measure(work)["S_mean"]

    kw = repair_kwargs(pc, inputs.rel_tol, row_set=row_set)
    kw.pop("damping"), kw.pop("gamma")
    repairers: dict = {}  # one per damping setting, reused across μ
    mus = args.mu or pc["mu_grid"]
    points = []
    for mu in mus:
        if modules is None:
            new, report, setting = trust_region_repair(
                inputs.fh,
                inputs.ft,
                W0,
                deltas,
                mu,
                gamma_c=pc["gamma_c"],
                gamma_e_grid=pc["gamma_e_grid"],
                actual_S=measured_S,
                repairers=repairers,
                **kw,
            )
        else:
            key = (pc["damping"], pc["gamma_c"])
            repairers.setdefault(key, ModelRepair(inputs.fh, inputs.ft, damping=key[0], gamma=key[1], **kw))
            new, report = repairers[key].repair(W0, deltas, mu)
            setting = {"damping": key[0], "gamma": key[1], "attempts": []}
        set_weights(work, theta1)  # modules outside `deltas` keep θ1's weights
        set_weights(work, new)
        measured = probe.measure(work)
        point = {
            "mu": mu,
            "damping": setting["damping"],
            "gamma": setting["gamma"],
            "trust_region": setting["attempts"],
            "trust_region_failed": bool(setting.get("failed")),
            "S_actual": measured["S"],
            "S_actual_mean": measured["S_mean"],
            "task_nll": measured["task_nll"],
            "task_regret_measured": measured["task_nll"] - drift["task_nll"],
            "S_pred": report["S_after_pred"],
            "S_before_pred": report["S_before_pred"],
            "R_pred": report["R_pred"],
            "max_step_ratio": report["max_step_ratio"],
            "max_input_step_ratio": report["max_input_step_ratio"],
            "step_ratios": {n: m["step_ratio"] for n, m in report["modules"].items()},
            "input_step_ratios": {n: m["input_step_ratio"] for n, m in report["modules"].items()},
        }
        if args.grad_diag:  # gradient at θ1, on `work`, over the repaired modules
            steps = repairers[(setting["damping"], setting["gamma"])].steps(deltas, mu, modules)
            set_weights(work, theta1)
            dev = encode_pairs(
                tok, [(r["prompt"], r["answer"]) for r in probe.task[:64]], ctx.cfg["factors"]["max_len"]
            )
            point["task_grad_diagnostic"] = task_grad_diagnostic(
                work, make_batches(dev, tok.pad_token_id, 1), steps, report["R_pred"]
            )
            set_weights(work, new)
        points.append(point)
        print(
            f"mu={mu:g} [{setting['damping']} {setting['gamma']:g}]: S {measured['S_mean']:.4g} "
            f"(drift {drift['S_mean']:.4g}, predicted {report['S_after_pred']:.3g}), "
            f"task NLL {measured['task_nll']:.3f}, "
            f"max step {report['max_step_ratio']:.2f} (input-weighted {report['max_input_step_ratio']:.2f})"
            + ("  TRUST REGION FAILED at every damping: do not evaluate this point" if setting.get("failed") else "")
        )
        if args.save and any(abs(mu - x) < 1e-12 for x in args.save):
            path = save_checkpoint(
                work, tok, ctx.out("ckpt", "repaired", f"{tag}_mu{mu:g}"), REPAIRED_LABEL.format(mu=mu, run=args.run)
            )
            print(f"  saved {path}")

    key = (pc["damping"], pc["gamma_c"])
    repairers.setdefault(key, ModelRepair(inputs.fh, inputs.ft, damping=key[0], gamma=key[1], **kw))
    try:
        from pencil.stats import spearman_bootstrap_ci

        rho = spearman_bootstrap_ci([p["S_pred"] for p in points], [p["S_actual_mean"] for p in points])
    except Exception as e:  # too few points for a CI
        rho = {"error": str(e)}
    name = f"repair_sweep_{tag}"
    path = write_result(
        name,
        {
            "points": points,
            "S_drift": drift["S"],
            "task_nll": {"theta0": nll_theta0, "theta1": drift["task_nll"]},
            "frontier_pred": repairers[key].frontier(deltas, mus),
            "spearman_pred_vs_actual_S": rho,
            "rel_tol": inputs.rel_tol,
            "null_floors": inputs.floors,
            "langs": inputs.langs,
            "row_set": row_set,
            "modules": modules,
            "frac": args.frac,
        },
        config={"pencil": pc},
        checkpoint=str(adapter_dir(ctx, args.run, args.frac)),
        out_dir=ctx.out("results"),
    )
    plot_sweep(ctx.out("figures") / f"{name}.png", points)
    print(f"-> {path}")


def plot_sweep(path, points):
    fig, ax = subplots(1, 2, figsize=(8, 3.2))
    for lang in points[0]["S_actual"]:
        ax[0].plot([p["task_nll"] for p in points], [p["S_actual"][lang] for p in points], "o-", label=lang)
    ax[0].set(xlabel="task NLL on dev (lower is better)", ylabel="measured S = KL(θ0‖θ) on dev")
    ax[0].set_title("measured frontier", fontsize=9)
    ax[0].legend(fontsize=7)
    ax[1].loglog([max(p["S_pred"], 1e-12) for p in points], [max(p["S_actual_mean"], 1e-12) for p in points], "o")
    ax[1].set(xlabel="predicted S (K-FAC, a ranking)", ylabel="measured S")
    ax[1].set_title("prediction vs measurement", fontsize=9)
    save(fig, path)


def forecast(ctx, args):
    """One streaming pass over each language's harmful factors, forecasting every checkpoint at once."""
    pc = ctx.cfg["pencil"]
    model0 = load_model(ctx.aligned, ctx)
    langs = args.langs or available_langs(ctx)
    row_set = pc.get("forecast_row_set", "all")
    checkpoints = {}
    for run in ctx.cfg["runs"]:
        for frac in (round(f * 100) for f in ctx.cfg["save_fractions"]):
            path = adapter_dir(ctx, run, frac)
            if path.exists():
                checkpoints[f"{run}|{frac}"] = lora_delta(path, lowrank=True)
            else:
                print(f"missing adapter {path}: run 03_finetune.py --run {run} first")
    results = ctx.out_root / "results"
    base = refusal_rates(results, "probe_theta0")
    rows = []
    for lang in langs:
        fh, _, _, _ = prepare_factors(model0, harm_factors(ctx, [lang]), harm_factors(ctx, [lang]), pc)
        for key, per_module in forecast_many(fh, checkpoints, row_set).items():
            run, frac = key.split("|")
            ev = refusal_rates(results, f"probe_run{run}_frac{frac}")
            drop = base[lang] - ev[lang] if base and ev and lang in base and lang in ev else None
            total = sum(per_module.values())
            rows.append(
                {
                    "run": run,
                    "frac": int(frac) / 100,
                    "lang": lang,
                    "forecast": total,
                    "refusal_drop": drop,
                    "per_module": {n: per_module[n] for n in layer_order(per_module)},
                }
            )
            print(
                f"{run} {int(frac) / 100:.2f} {lang}: forecast {total:.4g}"
                + (f", refusal drop {drop:+.3f}" if drop is not None else "")
            )
    pairs = [(r["forecast"], r["refusal_drop"]) for r in rows if r["refusal_drop"] is not None]
    rho = None
    if len(pairs) >= 4:
        from pencil.stats import spearman_bootstrap_ci

        rho = spearman_bootstrap_ci([a for a, _ in pairs], [b for _, b in pairs])
        print(f"Spearman(forecast, refusal drop) over {len(pairs)} checkpoint-language pairs: {rho}")
    write_result(
        "forecast_p2",
        {"rows": rows, "spearman": rho, "row_set": row_set, "langs": langs},
        config={"pencil": pc},
        out_dir=ctx.out("results"),
    )


def refusal_rates(results_dir, name):
    """{lang: refusal rate} from a 02_evaluate.py result, or None if it doesn't exist yet."""
    path = results_dir / f"{name}.json"
    if not path.exists():
        return None
    metrics = json.loads(path.read_text())["result"]["metrics"]
    return {lang: m["harmful"]["refusal"]["p"] for lang, m in metrics.items() if "harmful" in m}


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
    s.add_argument("--grad-diag", action="store_true", help="task-gradient diagnostic (needs weight gradients)")
    f = common_args(sub.add_parser("forecast"))
    f.add_argument("--langs", nargs="+")
    args = p.parse_args(argv)
    ctx = setup(args)
    (sweep if args.cmd == "sweep" else forecast)(ctx, args)


if __name__ == "__main__":
    main()
