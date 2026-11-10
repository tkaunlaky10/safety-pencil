"""Invariance demo (Prompt 5; prediction P1).

Reparameterise with a random V/O gauge (one head_dim × head_dim matrix per KV head and layer, condition
number 10 and 30), transform the fine-tune's update and the factors consistently, repair in both
coordinates, and compare the two repaired models' outputs (KL and max |Δ log-prob|; never raw weights).

Methods: the pencil (covariant damping: expected invariant), the pencil with Euclidean damping (expected
small non-zero), RegMean (invariant up to its pseudo-inverse cutoff), the diagonal Fisher merge,
AlignMerge-style, Safe LoRA (expected non-invariant, the negative control) and a linear ReSta without DARE
(the positive control; DARE's random element-wise drop is itself not gauge-invariant).

Only v_proj and o_proj are repaired by default (--modules vo; the others keep θ1's weights): the gauge
touches nothing else, so every other module would be repaired identically in both coordinates and can't
change the comparison, and the demo costs minutes instead of the full MLP-width eigensolves.

Precision: run it in float32 (the default here). In bf16 the gauge alone moves log-probs by ~1e-3, which
hides the difference between invariant and non-invariant methods. One model object is used: each weight
setting is loaded in turn and log-probabilities are compared against a CPU table, so Tiny Aya in fp32
(13 GB) fits an L4.

Also checks the analytic factor transformation against re-estimating the factors on the reparameterised
θ0 (same prompts, same sampling seed) on a few layers, and reports the conditioning of the gauge-touched
factors (numerical invariance degrades as their smallest eigenvalue ratio shrinks).

    python scripts/05_invariance.py --run A
    python scripts/05_invariance.py --run A --dry-run
"""
from __future__ import annotations

import argparse
import json

import torch

from pencil.baselines import diag_fisher_delta, euclidean_projection, kfac_diagonal, regmean_delta, safe_lora_delta
from pencil.delta import as_dense, lora_delta, set_weights
from pencil.factors import KFACCollector, LazyFactors, parse_name, target_modules
from pencil.generate import generate_cached
from pencil.invariance import (AttnShape, gauge_transform, kl_to_table, logprob_table, random_gauges,
                               reparam_weights, transform_delta)
from pencil.repair import ModelRepair, model_weights
from pencil.results import write_result
from pencil.runtime import (available_langs, common_args, encode_pairs, harm_factors, harmful_rows, load_factors,
                            load_model, load_tokenizer, make_batches, prepare_factors, repair_kwargs, setup,
                            task_factors)


def harm_batches(ctx, model0, tok, lang, split, n):
    gen = ctx.cfg["generation"]
    rows = harmful_rows(ctx, lang, split, n)
    key = f"{split}_harmful_{lang}"
    resp = generate_cached(model0, tok, {key: rows}, checkpoint=ctx.aligned, max_new_tokens=gen["max_new_tokens"],
                           batch_size=gen["batch_size"], root=ctx.out("generations"))[key]
    enc = encode_pairs(tok, [(r["prompt"], resp[r["id"]]) for r in rows], ctx.cfg["factors"]["max_len"])
    return make_batches(enc, tok.pad_token_id, 1)


def conditioning(fh, ft, modules, row_set) -> dict:
    """min/max eigenvalue of F_h + F_t on the two gauge-touched sides (o_proj input, v_proj output)."""
    out = {}
    rs = row_set if isinstance(row_set, str) else row_set.get("default", "all")
    for n in ModelRepair.order(modules):
        leaf = n.rsplit(".", 1)[-1]
        if leaf not in ("o_proj", "v_proj"):
            continue
        i = 0 if leaf == "o_proj" else 1
        F = (fh.pair(n, rs)[i] + ft.pair(n, rs)[i]).double()
        w = torch.linalg.eigvalsh(0.5 * (F + F.T))
        out[n] = float(w[0].clamp_min(0) / w[-1])
    return out


def reestimation_check(ctx, model, tok, gauges, shape, lang, layers, W0_vo):
    """Max and median relative difference between factors re-estimated on the reparameterised θ0 and the
    analytically transformed ones, per factor, scaled by the module's largest factor over row sets.
    `model` must hold θ0's weights on entry; it is restored on exit."""
    meta = json.loads((ctx.out_root / "factors" / "harm" / lang / "meta.json").read_text())
    batches = harm_batches(ctx, model, tok, lang, "factor", meta["n_sequences"])
    set_weights(model, reparam_weights(W0_vo, gauges, shape))
    store = load_factors(ctx, "harm", lang)
    with KFACCollector(model, layers=layers, row_sets=tuple(store.row_sets), seed=meta["seed"],
                       accum_dtype=getattr(torch, ctx.cfg["factors"].get("accum_dtype", "float32"))) as kc:
        for b in batches:
            kc.collect(b)
    f_re = kc.finalize()
    set_weights(model, W0_vo)
    f_an = LazyFactors([store]).with_transform(gauge_transform(gauges, shape))
    worst = {}
    for n in ModelRepair.order(f_re.modules):
        pairs = {rs: (f_re.pair(n, rs), f_an.pair(n, rs)) for rs in f_re.row_sets}
        scale = [max(float(y[i].norm()) for _, y in pairs.values()) for i in (0, 1)]
        for rs, (x, y) in pairs.items():
            for i in (0, 1):
                worst[f"{'AG'[i]}::{n}::{rs}"] = float((x[i].double() - y[i]).norm() / max(scale[i], 1e-30))
    vals = sorted(worst.values())
    return {"max_rel_diff": vals[-1], "median_rel_diff": vals[len(vals) // 2], "per_factor": worst, "layers": layers}


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
    tok = load_tokenizer(ctx)
    model = load_model(ctx.aligned, ctx, dtype=getattr(torch, args.dtype))
    shape = AttnShape.from_config(model.config)
    langs = available_langs(ctx)
    deltas = lora_delta(ctx.out_root / "ckpt" / args.run / "frac100", lowrank=True)
    fh, ft, tol, _ = prepare_factors(model, harm_factors(ctx, langs), task_factors(ctx, args.run), pc)
    vo_all = [n for n in target_modules(model) if parse_name(n)[2] in ("v_proj", "o_proj")]
    repaired = sorted(n for n in deltas if n in fh.a_key and (args.modules == "all" or n in vo_all))
    D = {n: deltas[n] for n in repaired}
    W0_all = model_weights(model, set(deltas) | set(vo_all), device="cpu")       # θ0, to restore
    W0_vo = {n: W0_all[n] for n in vo_all}
    W0 = {n: W0_all[n] for n in repaired}
    theta1 = {n: (W0_all[n].double() + as_dense(deltas[n], torch.float64)).to(W0_all[n].dtype) for n in deltas}
    base = load_model(ctx.base, ctx, dtype=torch.float32).cpu()
    Wb = model_weights(base, repaired, device="cpu")
    del base
    row_set = pc.get("repair_row_set", "all")
    rs1 = row_set if isinstance(row_set, str) else row_set.get("default", "all")
    batches = {"factor": harm_batches(ctx, model, tok, langs[0], "factor", args.n_prompts),
               "heldout": harm_batches(ctx, model, tok, langs[0], "dev", args.n_prompts)}
    sel = ctx.cfg["factors"]["selection_layers"]
    re_layers = [int(x) for x in args.reestimate_layers.split(",")] if args.reestimate_layers else \
        [l for l in sel if l < shape.n_layers][:2]
    kw = repair_kwargs(pc, tol, row_set=row_set)
    kw.pop("damping"), kw.pop("gamma")

    def pencil(damping, gamma, H, T, W, Dd):
        return ModelRepair(H, T, damping=damping, gamma=gamma, **kw).repair(W, Dd, args.mu)[0]

    def per_module(fn, H, T, W, Dd, Wbase):
        return {n: (W[n].double() + fn(n, as_dense(Dd[n], torch.float64), H, T, W[n].double(), Wbase[n].double())).to(W[n].dtype)
                for n in ModelRepair.order(Dd)}

    def am(H):
        return H.with_transform(lambda kind, key, F: F if kind == "D" else euclidean_projection(F, min(args.alignmerge_rank, F.shape[0])))

    methods = {
        "pencil_covariant": lambda H, T, W, Dd, B: pencil("covariant", pc["gamma_c"], H, T, W, Dd),
        "pencil_euclidean": lambda H, T, W, Dd, B: pencil("euclidean", args.gamma_e, H, T, W, Dd),
        "alignmerge_style": lambda H, T, W, Dd, B: pencil("covariant", pc["gamma_c"], am(H), T, W, Dd),
        "regmean": lambda H, T, W, Dd, B: per_module(
            lambda n, d, h, t, w, b: regmean_delta(d, h.pair(n, rs1)[0], t.pair(n, rs1)[0], args.mu_regmean, 1e-13),
            H, T, W, Dd, B),
        "diag_fisher": lambda H, T, W, Dd, B: per_module(
            lambda n, d, h, t, w, b: diag_fisher_delta(d, kfac_diagonal(*h.pair(n, rs1)), kfac_diagonal(*t.pair(n, rs1)),
                                                       args.mu), H, T, W, Dd, B),
        "safe_lora": lambda H, T, W, Dd, B: per_module(
            lambda n, d, h, t, w, b: safe_lora_delta(d, w, b, args.safelora_threshold)[0], H, T, W, Dd, B),
        "resta_linear": lambda H, T, W, Dd, B: per_module(
            lambda n, d, h, t, w, b: d + args.resta_alpha * (w - b), H, T, W, Dd, B),
    }

    results = {}
    for cond in args.conds:
        gauges = random_gauges(shape, cond=cond, seed=args.seed)
        W0r, Wbr, Dr = reparam_weights(W0, gauges, shape), reparam_weights(Wb, gauges, shape), transform_delta(D, gauges, shape)
        # θ1 in gauge coordinates: every v/o weight transformed (the gauge must cover all of them, repaired or not)
        theta1_r = {**theta1, **reparam_weights({n: theta1.get(n, W0_vo[n]) for n in vo_all}, gauges, shape)}
        g = gauge_transform(gauges, shape)
        fhr, ftr = fh.with_transform(g), ft.with_transform(g)
        res = {}
        for name, fn in methods.items():
            set_weights(model, theta1)
            set_weights(model, fn(fh, ft, W0, D, Wb))
            tables = {k: logprob_table(model, b) for k, b in batches.items()}
            set_weights(model, theta1_r)
            set_weights(model, fn(fhr, ftr, W0r, Dr, Wbr))
            res[name] = {k: kl_to_table(tables[k], model, b) for k, b in batches.items()}
            del tables
            print(f"cond {cond:g} {name:18s} factor-prompt max|Δlogp| {res[name]['factor']['max_abs_logprob_diff']:.2e} "
                  f"KL {res[name]['factor']['kl_mean']:.2e} | held-out {res[name]['heldout']['max_abs_logprob_diff']:.2e}")
        set_weights(model, W0_all)                                      # back to θ0 everywhere
        res["conditioning"] = conditioning(fh, ft, repaired, row_set)
        res["reestimation_check"] = reestimation_check(ctx, model, tok, gauges, shape, langs[0], re_layers, W0_vo)
        print(f"cond {cond:g}: worst min/max eigenvalue of a gauge-touched factor "
              f"{min(res['conditioning'].values()):.1e}; factor re-estimation vs analytic transform: max rel diff "
              f"{res['reestimation_check']['max_rel_diff']:.2e}, median {res['reestimation_check']['median_rel_diff']:.2e}")
        results[str(cond)] = res
    write_result(f"invariance_run{args.run}", {"results": results, "mu": args.mu, "gamma_e": args.gamma_e,
                                                "row_set": row_set, "rel_tol": tol, "dtype": args.dtype,
                                                "modules": args.modules},
                 config={"pencil": pc}, seed=args.seed, checkpoint=f"run{args.run}", out_dir=ctx.out("results"))
    plot(ctx, results)


def plot(ctx, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = list(results)
    names = [k for k, v in results[conds[0]].items() if isinstance(v, dict) and "factor" in v]
    fig, ax = plt.subplots(figsize=(7, 3))
    w = 0.8 / len(conds)
    for i, c in enumerate(conds):
        vals = [max(results[c][n]["factor"]["max_abs_logprob_diff"], 1e-12) for n in names]
        ax.bar([j + i * w for j in range(len(names))], vals, w, label=f"cond {c}")
    ax.set_yscale("log")
    ax.set_xticks([j + w * (len(conds) - 1) / 2 for j in range(len(names))], [n.replace("_", "\n") for n in names], fontsize=7)
    ax.set_ylabel("max |Δ log-prob|, original vs\nreparameterised repair")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ctx.out("figures") / "invariance.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
