"""Baselines on one run (Prompt 7), measured with the same cheap dev probe as the pencil sweep:
S = KL(θ0 ‖ θ) on θ0's dev answers to harmful prompts, per language, and the task NLL on dev.
Use --save to write merged checkpoints for 02_evaluate.py (refusal, over-refusal, MMLU with the judges).

Ours (same factors as the pencil, so the comparison isolates the curvature model):
  regmean          W1 − ΔW A_h (A_h + μA_t)⁺                         μ in [0.3, 1000], 13 log-spaced
  diag_fisher      per-weight Fisher merge (true diagonal if the factor pass ran with --diag, else
                   diag(G) diag(A)ᵀ, recorded)                       μ in [1e-3, 3], 11 log-spaced
  alignmerge       pencil with Euclidean rank-r harmful factors      r in {16, 64}, μ on the pencil grid
  fisher_standard  pencil with general-instruction factors at θ0 (needs factors/general/general)
  safelora         Safe LoRA projection C ΔW, C = VVᵀ/‖V‖, V = W_aligned − W_base (= projecting lora_B)
  safe_delta       SafeTune Safe Delta fed our A factors (hessian_inputs; our patch passes them through)
SafeTune (on the merged drifted model; base and aligned on CPU):
  wiseft, resta (DARE on), task_arithmetic, safemerge, lox, prepost (pre_model = tiny-aya-base)

    python scripts/07_baselines.py --run A --methods wiseft resta safelora safemerge regmean diag_fisher
    python scripts/07_baselines.py --run A --dry-run
"""
from __future__ import annotations

import argparse
import copy
import math

import torch

from pencil.baselines import (RegMean, alignmerge_transform, diag_fisher_delta, kfac_diagonal, safe_delta_hessians,
                              safe_lora_delta)
from pencil.delta import as_dense, lora_delta, merge_deltas, set_weights
from pencil.factors import LazyFactors
from pencil.evaluate import task_nll
from pencil.repair import ModelRepair, model_weights
from pencil.results import write_result
from pencil.runtime import (available_langs, common_args, dev_harm_batches, harm_factors, load_factors, load_model,
                            load_tokenizer, measure_S, prepare_factors, repair_kwargs, setup, task_dev_rows,
                            task_factors)

MUST = ["wiseft", "resta", "safelora", "safemerge", "regmean", "diag_fisher"]
NICE = ["alignmerge", "safe_delta", "task_arithmetic", "lox", "fisher_standard", "prepost"]


def logspace(a, b, n):
    return [10 ** (math.log10(a) + i * (math.log10(b) - math.log10(a)) / (n - 1)) for i in range(n)]


def main(argv=None):
    p = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--run", default="A")
    p.add_argument("--methods", nargs="+", default=MUST, help=f"must: {MUST}; nice: {NICE}")
    p.add_argument("--save", nargs="+", default=[], help="settings to save, e.g. regmean:3 wiseft:0.5")
    args = p.parse_args(argv)
    ctx = setup(args)
    pc = ctx.cfg["pencil"]
    tok = load_tokenizer(ctx)
    model0 = load_model(ctx.aligned, ctx)
    langs = available_langs(ctx)
    deltas = lora_delta(ctx.out_root / "ckpt" / args.run / "frac100", lowrank=True)
    fh, ft, tol, _ = prepare_factors(model0, harm_factors(ctx, langs), task_factors(ctx, args.run), pc)
    deltas = {n: deltas[n] for n in ModelRepair.order(n for n in deltas if n in fh.a_key)}
    rs = pc.get("repair_row_set", "all")
    rs1 = rs if isinstance(rs, str) else rs.get("default", "all")
    work = copy.deepcopy(model0)
    mods0 = dict(model0.named_modules())
    mods_w = dict(work.named_modules())

    def reset():
        """work ← θ1: every parameter from θ0 (SafeTune methods may touch embeddings and norms), then + ΔW."""
        with torch.no_grad():
            for (_, p_w), (_, p_0) in zip(work.named_parameters(), model0.named_parameters()):
                p_w.copy_(p_0)
        merge_deltas(work, deltas)

    def per_module(fn):
        """W0 + fn(n, ΔW) written straight into `work`, one module at a time (layer order, no weight dicts)."""
        with torch.no_grad():
            for n, d in deltas.items():
                W = mods0[n].weight
                new = W.double().cpu() + fn(n, as_dense(d, torch.float64)).cpu()
                mods_w[n].weight.copy_(new.to(W))

    reset()
    dev_h = dev_harm_batches(ctx, model0, tok, langs, ctx.cfg["evaluation"]["probe_per_lang"])
    task_dev = task_dev_rows(ctx)
    aligned_cpu = base_cpu = None

    def cpu_models():
        nonlocal aligned_cpu, base_cpu
        if aligned_cpu is None:                 # bf16 on CPU: 2 × 6.7 GB of RAM
            aligned_cpu = load_model(ctx.aligned, ctx, dtype=torch.bfloat16).cpu()
            base_cpu = load_model(ctx.base, ctx, dtype=torch.bfloat16).cpu()
        return aligned_cpu, base_cpu

    W0 = None

    def W0_cpu():
        nonlocal W0
        if W0 is None:
            W0 = model_weights(model0, deltas, device="cpu")
        return W0

    def settings(method):
        """Yield (setting, apply) pairs; apply() changes `work` in place (after reset())."""
        if method == "regmean":
            rm = RegMean(fh, ft, rs1, top_k=pc.get("top_k_mlp"), eig_device=repair_kwargs(pc, tol)["eig_device"])
            for mu in logspace(0.3, 1000, 13):
                yield mu, lambda mu=mu: per_module(lambda n, d: rm.delta(n, d, mu))
        elif method == "diag_fisher":
            exact = bool(fh.D and ft.D)
            print(f"diag_fisher: {'per-sequence diagonal from the factor pass' if exact else 'diag(G) diag(A)ᵀ fallback'}")

            def D_of(f, n):
                return f.get("D", n) if exact else kfac_diagonal(*f.pair(n, rs1))
            for mu in logspace(1e-3, 3, 11):
                yield mu, lambda mu=mu: per_module(lambda n, d: diag_fisher_delta(d, D_of(fh, n), D_of(ft, n), mu))
        elif method == "alignmerge":
            for r in (16, 64):
                rep = ModelRepair(fh.with_transform(alignmerge_transform(r)), ft, **repair_kwargs(pc, tol))
                for mu in pc["mu_grid"]:
                    yield f"r{r}_mu{mu:g}", lambda rep=rep, mu=mu: set_weights(work, rep.repair(W0_cpu(), deltas, mu)[0])
        elif method == "fisher_standard":
            fg, ft2, tol2, _ = prepare_factors(model0, LazyFactors([load_factors(ctx, "general", "general")]),
                                               task_factors(ctx, args.run), pc)
            rep = ModelRepair(fg, ft2, **repair_kwargs(pc, tol2))
            for mu in pc["mu_grid"]:
                yield mu, lambda mu=mu: set_weights(work, rep.repair(W0_cpu(), deltas, mu)[0])
        elif method == "safelora":
            al, ba = cpu_models()
            ma, mb = dict(al.named_modules()), dict(ba.named_modules())
            for th in (0.3, 0.5, 0.7):
                yield th, lambda th=th: per_module(
                    lambda n, d: safe_lora_delta(d, ma[n].weight, mb[n].weight, th)[0])
        elif method == "safe_delta":
            import safetune.recover as SR
            al, _ = cpu_models()
            print("safe_delta: W_sft comes from bf16 merged weights (SafeTune's interface), unlike the other methods")
            H = safe_delta_hessians(fh, rs1)          # materialises every module's A (~19 GB fp32 for 36 layers)
            for s_ in (0.05, 0.1, 0.2, 0.5):
                yield s_, lambda s_=s_: SR.apply_safe_delta(work, aligned=al, strength=s_, hessian_inputs=H,
                                                            param_filter=["layers."])
        else:
            import safetune.runner.recover as RR
            al, ba = cpu_models()
            grids = {
                "wiseft": ("WiseFTTrainer", "alpha", [0.25, 0.5, 0.75], {"aligned_model": al}),
                "resta": ("ReStaTrainer", "alpha", [0.1, 0.25, 0.5], {"base_model": ba, "aligned_model": al, "dare": True}),
                "task_arithmetic": ("TaskArithmeticTrainer", "alpha", [0.1, 0.25, 0.5], {"base_model": ba, "aligned_model": al}),
                "safemerge": ("SafeMergeTrainer", "threshold", [0.25, 0.35, 0.5], {"base_model": ba, "aligned_model": al}),
                "lox": ("LoXTrainer", "extrapolation_factor", [0.1, 0.3, 0.5], {"base_model": ba, "aligned_model": al}),
                "prepost": ("PrePostMergeTrainer", "alpha", [0.25, 0.5, 0.75], {"pre_model": ba}),
            }
            cls, key, grid, kw = grids[method]
            for v in grid:
                yield v, lambda v=v: getattr(RR, cls)(work, **kw, **{key: v}).apply()

    results = {"drift": {"S": measure_S(model0, work, dev_h), "task_nll": task_nll(work, tok, task_dev)["nll_per_token"]},
               "theta0": {"task_nll": task_nll(model0, tok, task_dev)["nll_per_token"]}}
    print(f"drift: S {results['drift']['S']}, task NLL {results['drift']['task_nll']:.3f}")
    for method in args.methods:
        rows = []
        try:
            for setting, apply in settings(method):
                reset()
                t0 = ctx.elapsed()
                apply()
                S = measure_S(model0, work, dev_h)
                t = task_nll(work, tok, task_dev)["nll_per_token"]
                rows.append({"setting": setting, "S": S, "S_mean": sum(S.values()) / len(S), "task_nll": t,
                             "seconds": ctx.elapsed() - t0})
                print(f"{method} {setting}: S {rows[-1]['S_mean']:.4g}, task NLL {t:.3f}")
                tag = f"{method}:{setting:g}" if isinstance(setting, float) else f"{method}:{setting}"
                if tag in args.save:
                    d = ctx.out("ckpt", "baselines", f"run{args.run}_{tag.replace(':', '_')}")
                    work.save_pretrained(d)
                    tok.save_pretrained(d)
                    (d / "RESEARCH_ARTIFACT.txt").write_text(f"research artifact: baseline {tag} on run {args.run}\n")
        except Exception as e:                      # one broken baseline must not stop the rest
            rows.append({"error": f"{type(e).__name__}: {e}"})
            print(f"{method}: FAILED {type(e).__name__}: {e}")
        results[method] = rows
    reset()
    write_result(f"baselines_run{args.run}", results, config={"pencil": pc, "methods": args.methods},
                 checkpoint=f"run{args.run}", out_dir=ctx.out("results"))


if __name__ == "__main__":
    main()
