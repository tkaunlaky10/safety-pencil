"""Baselines on one run (Prompt 7), measured with the same dev probe as the pencil sweep (drift S per language
and task NLL). --save writes merged checkpoints for 02_evaluate.py.

Ours (same factors as the pencil, so the comparison isolates the curvature model):
  regmean          W1 − ΔW A_h (A_h + μA_t)⁺                       μ in [0.3, 1000], 13 log-spaced
  diag_fisher      per-weight Fisher merge (per-sequence diagonal if the factor pass ran with --diag,
                   else diag(G) diag(A)ᵀ; printed)                 μ in [1e-3, 3], 11 log-spaced
  alignmerge       pencil with Euclidean rank-r harmful factors    r in {16, 64}, μ on the pencil grid
  fisher_standard  pencil with general-instruction factors at θ0 (needs factors/general/general)
  safelora         Safe LoRA projection C ΔW, C = VVᵀ/‖V‖, V = W_aligned − W_base
  safe_delta       SafeTune's Safe Delta fed our input factors
SafeTune's (on the merged drifted model; aligned and base models in bf16 on the CPU):
  wiseft, resta (DARE on), task_arithmetic, safemerge, lox, prepost (pre_model = tiny-aya-base)

    python scripts/07_baselines.py --run A --methods wiseft resta safelora safemerge regmean diag_fisher
    python scripts/07_baselines.py --run A --dry-run
"""

from __future__ import annotations

import argparse
import copy
import math

import torch

from pencil.baselines import (
    RegMean,
    alignmerge_transform,
    diag_fisher_delta,
    kfac_diagonal,
    safe_delta_hessians,
    safe_lora_delta,
)
from pencil.delta import as_dense, merge_deltas, model_weights, save_checkpoint, set_weights
from pencil.factor_store import LazyFactors
from pencil.repair import ModelRepair
from pencil.results import write_result
from pencil.runtime import (
    DevProbe,
    common_args,
    load_factors,
    load_model,
    load_repair_inputs,
    prepare_factors,
    repair_kwargs,
    setup,
    single_row_set,
    task_factors,
)

MUST = ["wiseft", "resta", "safelora", "safemerge", "regmean", "diag_fisher"]
NICE = ["alignmerge", "safe_delta", "task_arithmetic", "lox", "fisher_standard", "prepost"]
SAFETUNE = {  # alias: (trainer class, swept argument, grid, which reference models it takes)
    "wiseft": ("WiseFTTrainer", "alpha", [0.25, 0.5, 0.75], {"aligned_model"}),
    "resta": ("ReStaTrainer", "alpha", [0.1, 0.25, 0.5], {"base_model", "aligned_model"}),
    "task_arithmetic": ("TaskArithmeticTrainer", "alpha", [0.1, 0.25, 0.5], {"base_model", "aligned_model"}),
    "safemerge": ("SafeMergeTrainer", "threshold", [0.25, 0.35, 0.5], {"base_model", "aligned_model"}),
    "lox": ("LoXTrainer", "extrapolation_factor", [0.1, 0.3, 0.5], {"base_model", "aligned_model"}),
    "prepost": ("PrePostMergeTrainer", "alpha", [0.25, 0.5, 0.75], {"pre_model"}),
}


def logspace(a, b, n):
    return [10 ** (math.log10(a) + i * (math.log10(b) - math.log10(a)) / (n - 1)) for i in range(n)]


class Baselines:
    """Holds θ0, a working copy reset to θ1 before every setting, and the factors. `settings(method)` yields
    (setting, apply) pairs; apply() edits the working copy in place."""

    def __init__(self, ctx, run: str):
        self.ctx, self.run, self.pc = ctx, run, ctx.cfg["pencil"]
        self.inputs = load_repair_inputs(ctx, run)
        self.model0, self.deltas = self.inputs.model0, self.inputs.deltas
        self.work = copy.deepcopy(self.model0)
        self.row_set = single_row_set(self.pc)
        self._references = None

    def reset(self):
        """work ← θ1: every parameter from θ0 (SafeTune methods may touch embeddings and norms), then + ΔW."""
        with torch.no_grad():
            for p_work, p_0 in zip(self.work.parameters(), self.model0.parameters()):
                p_work.copy_(p_0)
        merge_deltas(self.work, self.deltas)

    def per_module(self, fn):
        """work ← W0 + fn(module, ΔW), one module at a time in layer order."""
        with torch.no_grad():
            for n, d in self.deltas.items():
                W0 = self.model0.get_submodule(n).weight
                self.work.get_submodule(n).weight.copy_(
                    (W0.double().cpu() + fn(n, as_dense(d, torch.float64)).cpu()).to(W0)
                )

    def references(self):
        """(aligned, base) models in bf16 on the CPU, loaded on first use."""
        if self._references is None:
            self._references = tuple(
                load_model(spec, self.ctx, dtype=torch.bfloat16).cpu() for spec in (self.ctx.aligned, self.ctx.base)
            )
        return self._references

    def pencil_like(self, fh, ft, rel_tol, tag_prefix=""):
        repairer = ModelRepair(fh, ft, **repair_kwargs(self.pc, rel_tol))
        W0 = model_weights(self.model0, self.deltas, device="cpu")
        for mu in self.pc["mu_grid"]:
            yield (
                f"{tag_prefix}{mu:g}" if tag_prefix else mu,
                lambda mu=mu: set_weights(self.work, repairer.repair(W0, self.deltas, mu)[0]),
            )

    def settings(self, method):
        fh, ft, rs = self.inputs.fh, self.inputs.ft, self.row_set
        if method == "regmean":
            regmean = RegMean(
                fh,
                ft,
                rs,
                top_k=self.pc.get("top_k_mlp"),
                eig_device=repair_kwargs(self.pc, self.inputs.rel_tol)["eig_device"],
            )
            for mu in logspace(0.3, 1000, 13):
                yield mu, lambda mu=mu: self.per_module(lambda n, d: regmean.delta(n, d, mu))
        elif method == "diag_fisher":
            exact = fh.has_diag and ft.has_diag
            print(
                f"diag_fisher: {'per-sequence diagonal from the factor pass' if exact else 'diag(G) diag(A)ᵀ fallback'}"
            )

            def diagonal(f, n):
                return f.get("D", n) if exact else kfac_diagonal(*f.pair(n, rs))

            for mu in logspace(1e-3, 3, 11):
                yield (
                    mu,
                    lambda mu=mu: self.per_module(
                        lambda n, d: diag_fisher_delta(d, diagonal(fh, n), diagonal(ft, n), mu)
                    ),
                )
        elif method == "alignmerge":
            for r in (16, 64):
                yield from self.pencil_like(
                    fh.with_transform(alignmerge_transform(r)), ft, self.inputs.rel_tol, f"r{r}_mu"
                )
        elif method == "fisher_standard":
            general = LazyFactors([load_factors(self.ctx, "general", "general")])
            fg, ft2, rel_tol, _ = prepare_factors(self.model0, general, task_factors(self.ctx, self.run), self.pc)
            yield from self.pencil_like(fg, ft2, rel_tol)
        elif method == "safelora":
            aligned, base = self.references()
            for th in (0.3, 0.5, 0.7):
                yield (
                    th,
                    lambda th=th: self.per_module(
                        lambda n, d: safe_lora_delta(
                            d, aligned.get_submodule(n).weight, base.get_submodule(n).weight, th
                        )[0]
                    ),
                )
        elif method == "safe_delta":
            import safetune.recover as recover

            aligned, _ = self.references()
            print("safe_delta: W_sft comes from bf16 merged weights (SafeTune's interface), unlike the other methods")
            hessians = safe_delta_hessians(fh, rs)  # every module's input factor (~19 GB fp32 for 36 layers)
            for s in (0.05, 0.1, 0.2, 0.5):
                yield (
                    s,
                    lambda s=s: recover.apply_safe_delta(
                        self.work, aligned=aligned, strength=s, hessian_inputs=hessians, param_filter=["layers."]
                    ),
                )
        else:
            import safetune.runner.recover as recover

            aligned, base = self.references()
            cls, arg, grid, refs = SAFETUNE[method]
            models = {"aligned_model": aligned, "base_model": base, "pre_model": base}
            kw = {k: models[k] for k in refs} | ({"dare": True} if method == "resta" else {})
            for v in grid:
                yield v, lambda v=v: getattr(recover, cls)(self.work, **kw, **{arg: v}).apply()


def main(argv=None):
    p = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--run", default="A")
    p.add_argument("--methods", nargs="+", default=MUST, help=f"must: {MUST}; nice: {NICE}")
    p.add_argument("--save", nargs="+", default=[], help="settings to save, e.g. regmean:3 wiseft:0.5")
    args = p.parse_args(argv)
    ctx = setup(args)
    b = Baselines(ctx, args.run)
    probe = DevProbe(ctx, b.model0, b.inputs.tok, b.inputs.langs)
    b.reset()
    results = {"drift": probe.measure(b.work), "theta0": {"task_nll": probe.task_nll(b.model0)}}
    print(f"drift: S {results['drift']['S']}, task NLL {results['drift']['task_nll']:.3f}")
    for method in args.methods:
        rows = []
        try:
            for setting, apply in b.settings(method):
                b.reset()
                t0 = ctx.elapsed()
                apply()
                rows.append({"setting": setting, **probe.measure(b.work), "seconds": ctx.elapsed() - t0})
                print(f"{method} {setting}: S {rows[-1]['S_mean']:.4g}, task NLL {rows[-1]['task_nll']:.3f}")
                tag = f"{method}:{setting:g}" if isinstance(setting, float) else f"{method}:{setting}"
                if tag in args.save:
                    save_checkpoint(
                        b.work,
                        b.inputs.tok,
                        ctx.out("ckpt", "baselines", f"run{args.run}_{tag.replace(':', '_')}"),
                        f"research artifact: baseline {tag} on run {args.run}",
                    )
        except Exception as e:  # one broken baseline must not stop the rest
            rows.append({"error": f"{type(e).__name__}: {e}"})
            print(f"{method}: FAILED {type(e).__name__}: {e}")
        results[method] = rows
    write_result(
        f"baselines_run{args.run}",
        results,
        config={"pencil": b.pc, "methods": args.methods},
        checkpoint=f"run{args.run}",
        out_dir=ctx.out("results"),
    )


if __name__ == "__main__":
    main()
