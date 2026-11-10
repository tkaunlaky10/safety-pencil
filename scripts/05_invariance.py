"""Invariance demo (Prompt 5; prediction P1).

Reparameterise with a random V/O gauge (one head_dim × head_dim matrix per KV head and layer, condition
number 10 and 30), transform the fine-tune's update and the factors consistently, repair in both
coordinates, and compare the two repaired models' outputs (KL and max |Δ log-prob|, never raw weights).

Expected: the covariant pencil and the linear ReSta (no DARE; the positive control) are invariant, RegMean
too up to its pseudo-inverse cutoff; Euclidean damping, the diagonal Fisher merge, AlignMerge-style and
Safe LoRA (the negative control) are not.

Only v_proj and o_proj are repaired by default (--modules vo); the gauge touches nothing else, so other
modules would be repaired identically in both coordinates. Runs in float32 (in bf16 the gauge alone moves
log-probs by ~1e-3) with one model object: each weight setting is loaded in turn and compared against a
CPU table of log-probs, so Tiny Aya in fp32 (13 GB) fits an L4.

Also checks the analytic factor transform against factors re-estimated on the reparameterised θ0, and
reports how well-conditioned the gauge-touched factors are (numerical invariance degrades with it).

    python scripts/05_invariance.py --run A
    python scripts/05_invariance.py --run A --dry-run
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import torch

from pencil.baselines import alignmerge_transform, diag_fisher_delta, kfac_diagonal, regmean_delta, safe_lora_delta
from pencil.delta import as_dense, model_weights, set_weights
from pencil.factor_store import LazyFactors
from pencil.factors import KFACCollector
from pencil.invariance import AttnShape, gauge_transform, gauge_weights, kl_to_table, logprob_table, random_gauges
from pencil.modules import layer_order, leaf, target_modules
from pencil.plots import save, subplots
from pencil.repair import ModelRepair
from pencil.results import write_result
from pencil.runtime import (
    common_args,
    load_factors,
    load_model,
    load_repair_inputs,
    repair_kwargs,
    setup,
    single_row_set,
    theta0_answers,
)

F64 = torch.float64


@dataclass
class Coords:
    """Everything a repair method needs, in one coordinate system."""

    fh: LazyFactors
    ft: LazyFactors
    W0: dict
    deltas: dict
    W_base: dict


def repair_methods(args, pc: dict, repair_kw: dict, row_set: str) -> dict:
    """name -> function(Coords) -> repaired weights {module: W0 + ΔW′}."""

    def per_module(c, fn):
        return {n: (c.W0[n].to(F64) + fn(n, as_dense(d, F64))).to(c.W0[n].dtype) for n, d in c.deltas.items()}

    def pencil(damping, gamma, transform=None):
        def run(c):
            fh = c.fh if transform is None else c.fh.with_transform(transform)
            return ModelRepair(fh, c.ft, damping=damping, gamma=gamma, **repair_kw).repair(c.W0, c.deltas, args.mu)[0]

        return run

    def A(f, n):
        return f.pair(n, row_set)[0]

    return {
        "pencil_covariant": pencil("covariant", pc["gamma_c"]),
        "pencil_euclidean": pencil("euclidean", args.gamma_e),
        "alignmerge_style": pencil("covariant", pc["gamma_c"], alignmerge_transform(args.alignmerge_rank)),
        "regmean": lambda c: per_module(
            c, lambda n, d: regmean_delta(d, A(c.fh, n), A(c.ft, n), args.mu_regmean, 1e-13)
        ),
        "diag_fisher": lambda c: per_module(
            c,
            lambda n, d: diag_fisher_delta(
                d, kfac_diagonal(*c.fh.pair(n, row_set)), kfac_diagonal(*c.ft.pair(n, row_set)), args.mu
            ),
        ),
        "safe_lora": lambda c: per_module(
            c, lambda n, d: safe_lora_delta(d, c.W0[n], c.W_base[n], args.safelora_threshold)[0]
        ),
        "resta_linear": lambda c: per_module(
            c, lambda n, d: d + args.resta_alpha * (c.W0[n].to(F64) - c.W_base[n].to(F64))
        ),
    }


def conditioning(fh, ft, modules, row_set: str) -> dict[str, float]:
    """min/max eigenvalue of F_h + F_t on the two gauge-touched sides (o_proj input, v_proj output)."""
    out = {}
    for n in layer_order(m for m in modules if leaf(m) in ("o_proj", "v_proj")):
        side = 0 if leaf(n) == "o_proj" else 1
        w = torch.linalg.eigvalsh(fh.pair(n, row_set)[side] + ft.pair(n, row_set)[side])
        out[n] = float(w[0].clamp_min(0) / w[-1])
    return out


def reestimation_check(ctx, model, tok, gauges, shape, lang, layers, W0_vo) -> dict:
    """Factors re-estimated on the reparameterised θ0 (same prompts and sampling seed as the factor pass) vs
    the analytic transform: relative difference per factor, scaled by the module's largest factor over row
    sets (k_proj's full_reduce G is ~0 on full-attention layers). `model` must hold θ0 and is restored."""
    meta = json.loads((ctx.out_root / "factors" / "harm" / lang / "meta.json").read_text())
    batches = theta0_answers(
        ctx, model, tok, lang, "factor", meta["n_sequences"], ctx.cfg["factors"].get("batch_size", 1)
    )
    store = load_factors(ctx, "harm", lang)
    set_weights(model, gauge_weights(W0_vo, gauges, shape))
    with KFACCollector(
        model,
        layers=layers,
        row_sets=tuple(store.row_sets),
        seed=meta["seed"],
        accum_dtype=getattr(torch, ctx.cfg["factors"].get("accum_dtype", "float32")),
    ) as kc:
        for b in batches:
            kc.collect(b)
    set_weights(model, W0_vo)
    estimated, analytic = kc.finalize(), LazyFactors([store]).with_transform(gauge_transform(gauges, shape))
    diffs = {}
    for n in layer_order(estimated.modules):
        pairs = {rs: (estimated.pair(n, rs), analytic.pair(n, rs)) for rs in estimated.row_sets}
        scale = [max(float(y[i].norm()) for _, y in pairs.values()) for i in (0, 1)]
        for rs, (x, y) in pairs.items():
            for i, kind in enumerate("AG"):
                diffs[f"{kind}::{n}::{rs}"] = float((x[i].to(F64) - y[i]).norm() / max(scale[i], 1e-30))
    values = sorted(diffs.values())
    return {
        "max_rel_diff": values[-1],
        "median_rel_diff": values[len(values) // 2],
        "per_factor": diffs,
        "layers": layers,
    }


def main(argv=None):
    p = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--run", default="A")
    p.add_argument("--conds", type=float, nargs="+", default=[10.0, 30.0])
    p.add_argument("--mu", type=float, default=0.1, help="pencil / AlignMerge μ")
    p.add_argument("--mu-regmean", type=float, default=3.0)
    p.add_argument("--gamma-e", type=float, default=1e-2)
    p.add_argument("--alignmerge-rank", type=int, default=64)
    p.add_argument("--resta-alpha", type=float, default=0.25)
    p.add_argument("--safelora-threshold", type=float, default=0.5)
    p.add_argument("--modules", choices=["vo", "all"], default="vo")
    p.add_argument("--n-prompts", type=int, default=16, help="prompts per comparison set (log-prob tables live in RAM)")
    p.add_argument("--dtype", default="float32", choices=["float32", "float64", "bfloat16"])
    p.add_argument("--reestimate-layers", default=None, help="comma list (default: first two selection layers)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    ctx = setup(args)
    pc = ctx.cfg["pencil"]
    inputs = load_repair_inputs(ctx, args.run, dtype=getattr(torch, args.dtype))
    model, tok = inputs.model0, inputs.tok
    shape = AttnShape.from_config(model.config)
    vo = [n for n in target_modules(model) if leaf(n) in ("v_proj", "o_proj")]
    deltas = {n: d for n, d in inputs.deltas.items() if args.modules == "all" or n in vo}

    W0_all = model_weights(model, set(inputs.deltas) | set(vo), device="cpu")  # θ0, to restore later
    W0_vo = {n: W0_all[n] for n in vo}
    theta1 = {n: (W0_all[n].to(F64) + as_dense(d, F64)).to(W0_all[n].dtype) for n, d in inputs.deltas.items()}
    W_base = model_weights(load_model(ctx.base, ctx, dtype=torch.float32).cpu(), deltas)
    row_set = single_row_set(pc)
    batches = {
        "factor": theta0_answers(ctx, model, tok, inputs.langs[0], "factor", args.n_prompts),
        "heldout": theta0_answers(ctx, model, tok, inputs.langs[0], "dev", args.n_prompts),
    }
    re_layers = (
        [int(x) for x in args.reestimate_layers.split(",")]
        if args.reestimate_layers
        else [layer for layer in ctx.cfg["factors"]["selection_layers"] if layer < shape.n_layers][:2]
    )
    # each method sets its own damping and γ
    repair_kw = {
        k: v for k, v in repair_kwargs(pc, inputs.rel_tol, row_set=row_set).items() if k not in ("damping", "gamma")
    }
    methods = repair_methods(args, pc, repair_kw, row_set)
    original = Coords(inputs.fh, inputs.ft, {n: W0_all[n] for n in deltas}, deltas, W_base)

    results = {}
    for cond in args.conds:
        gauges = random_gauges(shape, cond=cond, seed=args.seed)
        g = gauge_transform(gauges, shape)
        gauged = Coords(
            inputs.fh.with_transform(g),
            inputs.ft.with_transform(g),
            gauge_weights(original.W0, gauges, shape),
            gauge_weights(deltas, gauges, shape),
            gauge_weights(W_base, gauges, shape),
        )
        # θ1 in gauge coordinates: every v/o weight transformed, repaired or not
        theta1_gauged = theta1 | gauge_weights({n: theta1.get(n, W0_vo[n]) for n in vo}, gauges, shape)
        res = {}
        for name, method in methods.items():
            set_weights(set_weights(model, theta1), method(original))
            tables = {k: logprob_table(model, b) for k, b in batches.items()}
            set_weights(set_weights(model, theta1_gauged), method(gauged))
            res[name] = {k: kl_to_table(tables[k], model, b) for k, b in batches.items()}
            factor, heldout = res[name]["factor"], res[name]["heldout"]
            print(
                f"cond {cond:g} {name:18s} factor prompts: max|Δlogp| {factor['max_abs_logprob_diff']:.2e} "
                f"KL {factor['kl_mean']:.2e} | held-out: {heldout['max_abs_logprob_diff']:.2e}"
            )
        set_weights(model, W0_all)
        res["conditioning"] = conditioning(inputs.fh, inputs.ft, deltas, row_set)
        res["reestimation_check"] = reestimation_check(
            ctx, model, tok, gauges, shape, inputs.langs[0], re_layers, W0_vo
        )
        check, worst = res["reestimation_check"], min(res["conditioning"].values())
        print(
            f"cond {cond:g}: worst min/max eigenvalue of a gauge-touched factor {worst:.1e}; "
            f"re-estimated vs analytic factors: max rel diff {check['max_rel_diff']:.2e}, "
            f"median {check['median_rel_diff']:.2e}"
        )
        results[str(cond)] = res
    write_result(
        f"invariance_run{args.run}",
        {
            "results": results,
            "mu": args.mu,
            "gamma_e": args.gamma_e,
            "row_set": row_set,
            "rel_tol": inputs.rel_tol,
            "dtype": args.dtype,
            "modules": args.modules,
        },
        config={"pencil": pc},
        seed=args.seed,
        checkpoint=f"run{args.run}",
        out_dir=ctx.out("results"),
    )
    plot(ctx.out("figures") / "invariance.png", results, list(methods))


def plot(path, results: dict, names: list[str]):
    fig, ax = subplots(figsize=(7, 3))
    width = 0.8 / len(results)
    for i, (cond, res) in enumerate(results.items()):
        ax.bar(
            [j + i * width for j in range(len(names))],
            [max(res[n]["factor"]["max_abs_logprob_diff"], 1e-12) for n in names],
            width,
            label=f"cond {cond}",
        )
    ax.set_yscale("log")
    ax.set_xticks(
        [j + width * (len(results) - 1) / 2 for j in range(len(names))],
        [n.replace("_", "\n") for n in names],
        fontsize=7,
    )
    ax.set_ylabel("max |Δ log-prob|, original vs\nreparameterised repair")
    ax.legend(fontsize=7)
    save(fig, path)


if __name__ == "__main__":
    main()
