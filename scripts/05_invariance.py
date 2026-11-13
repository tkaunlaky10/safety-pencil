"""Invariance demo (Prompt 5; prediction P1).

Reparameterise with a random V/O gauge (one head_dim × head_dim matrix per KV head and layer, condition
number 10 and 30), transform the fine-tune's update and the factors consistently, repair in both
coordinates, and compare the two repaired models' outputs (KL and max |Δ log-prob|, never raw weights).

Expected: the covariant pencil, the scaled pencil (13 Nov; ν damping on output sides except v_proj, per-module step
scaling) and the linear ReSta (no DARE; the positive control) are invariant, RegMean
too up to its pseudo-inverse cutoff; Euclidean damping, the diagonal Fisher merge, AlignMerge-style and
Safe LoRA (the negative control) are not.

Only v_proj and o_proj are repaired by default (--modules vo); the gauge touches nothing else, so other
modules would be repaired identically in both coordinates. Runs in float32 (in bf16 the gauge alone moves
log-probs by ~1e-3) with one model object: each weight setting is loaded in turn and compared against a
CPU table of log-probs, so Tiny Aya in fp32 (13 GB) fits an L4.

Also checks the analytic factor transform against factors re-estimated on the reparameterised θ0, and
reports how well-conditioned the gauge-touched factors are (numerical invariance degrades with it).

    python scripts/05_invariance.py --run A
    python scripts/05_invariance.py --run A --methods pencil_scaled pencil_covariant --out-name invariance_scaled_runA
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
    RepairInputs,
    common_args,
    factor_dir,
    load_factors,
    load_model,
    load_repair_inputs,
    repair_kwargs,
    scaled_repair_kwargs,
    setup,
    single_row_set,
    theta0_answers,
)

F64 = torch.float64
VO_LEAVES = ("v_proj", "o_proj")  # the modules the gauge touches
INPUT_SIDE, OUTPUT_SIDE = 0, 1  # positions of A and G in factors.pair(...)
FACTOR_KINDS = "AG"
REGMEAN_RCOND = 1e-13  # RegMean's pseudo-inverse cutoff
SCALE_FLOOR = 1e-30  # guards relative differences against all-zero factors
LOG_PLOT_FLOOR = 1e-12  # zero differences are drawn here on the log axis
BAR_GROUP_WIDTH = 0.8


@dataclass
class Coords:
    """Everything a repair method needs, in one coordinate system."""

    harm: LazyFactors
    task: LazyFactors
    W0: dict
    deltas: dict
    W_base: dict


class RepairMethods:
    """The repairs compared in both coordinate systems. Each method takes a Coords and returns the repaired
    weights {module: W0 + ΔW′} in W0's dtype."""

    def __init__(self, args, pencil_cfg: dict, repair_kw: dict, row_set: str, scaled_kw: dict):
        self.args = args
        self.pencil_cfg = pencil_cfg
        self.repair_kw = repair_kw  # ModelRepair arguments other than damping and γ
        self.row_set = row_set
        self.scaled_kw = scaled_kw  # ModelRepair arguments of the scaled repair

    def by_name(self) -> dict:
        """{name: method}, in the order the results and the figure list them."""
        return {
            "pencil_covariant": self.pencil_covariant,
            "pencil_scaled": self.pencil_scaled,
            "pencil_euclidean": self.pencil_euclidean,
            "alignmerge_style": self.alignmerge_style,
            "regmean": self.regmean,
            "diag_fisher": self.diag_fisher,
            "safe_lora": self.safe_lora,
            "resta_linear": self.resta_linear,
        }

    # -- the pencil and its variants
    def _pencil(self, coords: Coords, harm, damping: str, gamma: float) -> dict:
        repairer = ModelRepair(harm, coords.task, damping=damping, gamma=gamma, **self.repair_kw)
        repaired, _ = repairer.repair(coords.W0, coords.deltas, self.args.mu)
        return repaired

    def pencil_covariant(self, coords: Coords) -> dict:
        return self._pencil(coords, coords.harm, "covariant", self.pencil_cfg["gamma_c"])

    def pencil_scaled(self, coords: Coords) -> dict:
        repairer = ModelRepair(coords.harm, coords.task, **self.scaled_kw)
        repaired, _ = repairer.repair(coords.W0, coords.deltas, self.args.mu)
        return repaired

    def pencil_euclidean(self, coords: Coords) -> dict:
        return self._pencil(coords, coords.harm, "euclidean", self.args.gamma_e)

    def alignmerge_style(self, coords: Coords) -> dict:
        """The covariant pencil with the harmful factors replaced by their Euclidean rank-r projections."""
        harm = coords.harm.with_transform(alignmerge_transform(self.args.alignmerge_rank))
        return self._pencil(coords, harm, "covariant", self.pencil_cfg["gamma_c"])

    # -- per-module merges
    def _per_module(self, coords: Coords, new_delta) -> dict:
        """{module: W0 + new_delta(coords, module, ΔW)}, added in float64 and stored in W0's dtype."""
        repaired = {}
        for module, delta in coords.deltas.items():
            W0 = coords.W0[module]
            repaired[module] = (W0.to(F64) + new_delta(coords, module, as_dense(delta, F64))).to(W0.dtype)
        return repaired

    def regmean(self, coords: Coords) -> dict:
        return self._per_module(coords, self._regmean_delta)

    def _regmean_delta(self, coords: Coords, module: str, dW: torch.Tensor) -> torch.Tensor:
        """ΔW′ = ΔW − ΔW A_h (A_h + μA_t)⁺."""
        A_h = coords.harm.pair(module, self.row_set)[INPUT_SIDE]
        A_t = coords.task.pair(module, self.row_set)[INPUT_SIDE]
        return regmean_delta(dW, A_h, A_t, self.args.mu_regmean, REGMEAN_RCOND)

    def diag_fisher(self, coords: Coords) -> dict:
        return self._per_module(coords, self._diag_fisher_delta)

    def _diag_fisher_delta(self, coords: Coords, module: str, dW: torch.Tensor) -> torch.Tensor:
        """Per-weight Fisher merge with the K-FAC diagonals diag(G) diag(A)ᵀ."""
        D_h = kfac_diagonal(*coords.harm.pair(module, self.row_set))
        D_t = kfac_diagonal(*coords.task.pair(module, self.row_set))
        return diag_fisher_delta(dW, D_h, D_t, self.args.mu)

    def safe_lora(self, coords: Coords) -> dict:
        return self._per_module(coords, self._safe_lora_delta)

    def _safe_lora_delta(self, coords: Coords, module: str, dW: torch.Tensor) -> torch.Tensor:
        """Safe LoRA projection C ΔW with V = W0 − W_base."""
        projected, _ = safe_lora_delta(dW, coords.W0[module], coords.W_base[module], self.args.safelora_threshold)
        return projected

    def resta_linear(self, coords: Coords) -> dict:
        return self._per_module(coords, self._resta_delta)

    def _resta_delta(self, coords: Coords, module: str, dW: torch.Tensor) -> torch.Tensor:
        """Linear ReSta (no DARE): ΔW′ = ΔW + α (W0 − W_base)."""
        return dW + self.args.resta_alpha * (coords.W0[module].to(F64) - coords.W_base[module].to(F64))


@dataclass
class Experiment:
    """Everything the comparison at each condition number reuses. `inputs.model0` is the one model object."""

    inputs: RepairInputs
    shape: AttnShape
    vo_modules: list[str]  # every v_proj and o_proj in the model
    W0_all: dict  # θ0 for every fine-tuned or V/O module, to restore the model
    theta1: dict  # θ1 for every fine-tuned module
    row_set: str
    batches: dict  # {"factor": ..., "heldout": ...}: θ0's answers the outputs are compared on
    reestimate_layers: list[int]
    methods: dict  # name -> function(Coords) -> repaired weights
    original: Coords

    @property
    def W0_vo(self) -> dict:
        return {module: self.W0_all[module] for module in self.vo_modules}


def modules_to_repair(all_deltas: dict, vo_modules: list[str], which: str) -> dict:
    """ΔW of the modules to repair: v_proj and o_proj only (--modules vo) or every module (all)."""
    if which == "all":
        return dict(all_deltas)
    return {module: delta for module, delta in all_deltas.items() if module in vo_modules}


def fine_tuned_weights(base_weights: dict, deltas: dict) -> dict:
    """θ1 = W0 + ΔW for every module, added in float64 and stored in the base weights' dtype."""
    theta1 = {}
    for module, delta in deltas.items():
        W0 = base_weights[module]
        theta1[module] = (W0.to(F64) + as_dense(delta, F64)).to(W0.dtype)
    return theta1


def base_model_weights(ctx, modules) -> dict:
    """The base (pre-alignment) model's weights for `modules`, in float32 on the CPU. The model is loaded in bf16
    on the CPU and only these weights are converted: a whole fp32 copy next to θ0 overflowed an L4 job's 30 GB
    of RAM (pencil A was OOMKilled here on 12 Nov)."""
    base_model = load_model(ctx.base, ctx, dtype=torch.bfloat16, device="cpu")
    weights = {name: W.float() for name, W in model_weights(base_model, modules).items()}
    del base_model
    return weights


def repair_kwargs_without_damping(pencil_cfg: dict, rel_tol: float, row_set: str) -> dict:
    """ModelRepair arguments from the config, minus damping and γ, which each method sets itself."""
    kwargs = repair_kwargs(pencil_cfg, rel_tol, row_set=row_set)
    kwargs.pop("damping", None)
    kwargs.pop("gamma", None)
    return kwargs


def layers_to_reestimate(ctx, reestimate_layers: str | None, n_layers: int) -> list[int]:
    """--reestimate-layers, or by default the first two selection layers that exist in the model."""
    if reestimate_layers:
        return [int(layer) for layer in reestimate_layers.split(",")]
    existing = [layer for layer in ctx.cfg["factors"]["selection_layers"] if layer < n_layers]
    return existing[:2]


def prepare_experiment(ctx, args) -> Experiment:
    """Load θ0 in --dtype, its factors and the fine-tune, build θ1, and take θ0's answers to compare on."""
    pencil_cfg = ctx.cfg["pencil"]
    inputs = load_repair_inputs(ctx, args.run, dtype=getattr(torch, args.dtype))
    model = inputs.model0
    shape = AttnShape.from_config(model.config)
    vo_modules = [module for module in target_modules(model) if leaf(module) in VO_LEAVES]
    deltas = modules_to_repair(inputs.deltas, vo_modules, args.modules)

    W0_all = model_weights(model, set(inputs.deltas) | set(vo_modules), device="cpu")
    theta1 = fine_tuned_weights(W0_all, inputs.deltas)
    W_base = base_model_weights(ctx, deltas)
    row_set = single_row_set(pencil_cfg)
    lang = inputs.langs[0]
    batches = {
        "factor": theta0_answers(ctx, model, inputs.tokenizer, lang, "factor", args.n_prompts),
        "heldout": theta0_answers(ctx, model, inputs.tokenizer, lang, "dev", args.n_prompts),
    }
    repair_kw = repair_kwargs_without_damping(pencil_cfg, inputs.rel_tol, row_set)
    W0_repaired = {module: W0_all[module] for module in deltas}

    return Experiment(
        inputs=inputs,
        shape=shape,
        vo_modules=vo_modules,
        W0_all=W0_all,
        theta1=theta1,
        row_set=row_set,
        batches=batches,
        reestimate_layers=layers_to_reestimate(ctx, args.reestimate_layers, shape.n_layers),
        methods=chosen_methods(args, pencil_cfg, repair_kw, row_set, inputs.rel_tol),
        original=Coords(inputs.harm_factors, inputs.task_factors, W0_repaired, deltas, W_base),
    )


def chosen_methods(args, pencil_cfg: dict, repair_kw: dict, row_set: str, rel_tol: float) -> dict:
    """{name: method} for --methods (default: all), in the order RepairMethods lists them."""
    scaled_kw = scaled_repair_kwargs(pencil_cfg, rel_tol, row_set=row_set)
    every_method = RepairMethods(args, pencil_cfg, repair_kw, row_set, scaled_kw).by_name()
    if not args.methods:
        return every_method
    unknown = set(args.methods) - set(every_method)
    if unknown:
        raise SystemExit(f"unknown --methods {sorted(unknown)}; choose from {list(every_method)}")
    return {name: method for name, method in every_method.items() if name in args.methods}


# ---------------------------------------------------------------- one condition number


def gauge_coordinates(original: Coords, gauges, shape: AttnShape) -> Coords:
    """The same repair inputs after the V/O reparameterisation: factors transformed analytically, weights,
    deltas and base weights transformed like weights."""
    transform = gauge_transform(gauges, shape)
    return Coords(
        harm=original.harm.with_transform(transform),
        task=original.task.with_transform(transform),
        W0=gauge_weights(original.W0, gauges, shape),
        deltas=gauge_weights(original.deltas, gauges, shape),
        W_base=gauge_weights(original.W_base, gauges, shape),
    )


def gauged_theta1(experiment: Experiment, gauges) -> dict:
    """θ1 in gauge coordinates: every v_proj and o_proj weight is transformed, fine-tuned or not."""
    W0_vo = experiment.W0_vo
    vo_weights = {}
    for module in experiment.vo_modules:
        if module in experiment.theta1:
            vo_weights[module] = experiment.theta1[module]
        else:
            vo_weights[module] = W0_vo[module]
    return experiment.theta1 | gauge_weights(vo_weights, gauges, experiment.shape)


def compare_method(experiment: Experiment, method, gauged: Coords, theta1_gauged: dict) -> dict:
    """Repair in the original coordinates, then in gauge coordinates, and compare the two models' outputs on
    each batch set: {"factor": {...}, "heldout": {...}} with KL and max |Δ log-prob|."""
    model = experiment.inputs.model0
    set_weights(model, experiment.theta1)
    set_weights(model, method(experiment.original))
    tables = {}
    for name, batches in experiment.batches.items():
        tables[name] = logprob_table(model, batches)

    set_weights(model, theta1_gauged)
    set_weights(model, method(gauged))
    comparison = {}
    for name, batches in experiment.batches.items():
        comparison[name] = kl_to_table(tables[name], model, batches)
    return comparison


def conditioning(harm, task, modules, row_set: str) -> dict[str, float]:
    """min/max eigenvalue of F_h + F_t on the two gauge-touched sides (o_proj input, v_proj output)."""
    gauge_touched = [module for module in modules if leaf(module) in ("o_proj", "v_proj")]
    ratios = {}
    for module in layer_order(gauge_touched):
        side = INPUT_SIDE if leaf(module) == "o_proj" else OUTPUT_SIDE
        combined = harm.pair(module, row_set)[side] + task.pair(module, row_set)[side]
        eigenvalues = torch.linalg.eigvalsh(combined)
        ratios[module] = float(eigenvalues[0].clamp_min(0) / eigenvalues[-1])
    return ratios


def reestimated_and_analytic_factors(ctx, model, tokenizer, gauges, shape, lang: str, layers, W0_vo: dict):
    """(re-estimated, analytic) harmful factors in gauge coordinates. The first are collected on the
    reparameterised θ0 with the factor pass's prompts and sampling seed; the second are the stored factors
    under the analytic transform. `model` must hold θ0 and is restored."""
    meta = json.loads((factor_dir(ctx, "harm", lang) / "meta.json").read_text())
    factor_cfg = ctx.cfg["factors"]
    batch_size = factor_cfg.get("batch_size", 1)
    batches = theta0_answers(ctx, model, tokenizer, lang, "factor", meta["n_sequences"], batch_size)
    store = load_factors(ctx, "harm", lang)

    set_weights(model, gauge_weights(W0_vo, gauges, shape))
    with KFACCollector(
        model,
        layers=layers,
        row_sets=tuple(store.row_sets),
        seed=meta["seed"],
        accum_dtype=getattr(torch, factor_cfg.get("accum_dtype", "float32")),
    ) as collector:
        for batch in batches:
            collector.collect(batch)
    set_weights(model, W0_vo)

    estimated = collector.finalize()
    analytic = LazyFactors([store]).with_transform(gauge_transform(gauges, shape))
    return estimated, analytic


def relative_factor_differences(estimated, analytic) -> dict[str, float]:
    """‖re-estimated − analytic‖ per factor, keyed 'A::module::row_set' or 'G::module::row_set'. Each is divided
    by the module's largest analytic factor of that kind over row sets, because some row sets' factors are ~0
    (k_proj's full_reduce G on full-attention layers)."""
    differences = {}
    for module in layer_order(estimated.modules):
        pairs = {}
        for row_set in estimated.row_sets:
            pairs[row_set] = (estimated.pair(module, row_set), analytic.pair(module, row_set))
        scales = []
        for side in (INPUT_SIDE, OUTPUT_SIDE):
            scales.append(max(float(analytic_pair[side].norm()) for _, analytic_pair in pairs.values()))
        for row_set, (estimated_pair, analytic_pair) in pairs.items():
            for side, kind in enumerate(FACTOR_KINDS):
                difference = (estimated_pair[side].to(F64) - analytic_pair[side]).norm()
                differences[f"{kind}::{module}::{row_set}"] = float(difference / max(scales[side], SCALE_FLOOR))
    return differences


def reestimation_check(ctx, model, tokenizer, gauges, shape, lang: str, layers, W0_vo: dict) -> dict:
    """How far factors re-estimated on the reparameterised θ0 are from the analytic transform: the largest and
    median relative difference, and every factor's value."""
    estimated, analytic = reestimated_and_analytic_factors(ctx, model, tokenizer, gauges, shape, lang, layers, W0_vo)
    differences = relative_factor_differences(estimated, analytic)
    values = sorted(differences.values())
    return {
        "max_rel_diff": values[-1],
        "median_rel_diff": values[len(values) // 2],
        "per_factor": differences,
        "layers": layers,
    }


def print_method_line(cond: float, name: str, comparison: dict) -> None:
    factor = comparison["factor"]
    heldout = comparison["heldout"]
    print(
        f"cond {cond:g} {name:18s} factor prompts: max|Δlogp| {factor['max_abs_logprob_diff']:.2e} "
        f"KL {factor['kl_mean']:.2e} | held-out: {heldout['max_abs_logprob_diff']:.2e}"
    )


def compare_at_condition(ctx, args, experiment: Experiment, cond: float) -> dict:
    """One random gauge of condition number `cond`: every method's output comparison, the conditioning of the
    gauge-touched factors and the re-estimation check."""
    shape = experiment.shape
    inputs = experiment.inputs
    gauges = random_gauges(shape, cond=cond, seed=args.seed)
    gauged = gauge_coordinates(experiment.original, gauges, shape)
    theta1_gauged = gauged_theta1(experiment, gauges)

    results = {}
    for name, method in experiment.methods.items():
        results[name] = compare_method(experiment, method, gauged, theta1_gauged)
        print_method_line(cond, name, results[name])

    set_weights(inputs.model0, experiment.W0_all)
    results["conditioning"] = conditioning(
        inputs.harm_factors, inputs.task_factors, experiment.original.deltas, experiment.row_set
    )
    results["reestimation_check"] = reestimation_check(
        ctx,
        inputs.model0,
        inputs.tokenizer,
        gauges,
        shape,
        inputs.langs[0],
        experiment.reestimate_layers,
        experiment.W0_vo,
    )
    check = results["reestimation_check"]
    worst = min(results["conditioning"].values())
    print(
        f"cond {cond:g}: worst min/max eigenvalue of a gauge-touched factor {worst:.1e}; "
        f"re-estimated vs analytic factors: max rel diff {check['max_rel_diff']:.2e}, "
        f"median {check['median_rel_diff']:.2e}"
    )
    return results


# ---------------------------------------------------------------- output


def write_invariance_result(ctx, args, experiment: Experiment, results: dict) -> None:
    write_result(
        args.out_name or f"invariance_run{args.run}",
        {
            "results": results,
            "mu": args.mu,
            "gamma_e": args.gamma_e,
            "row_set": experiment.row_set,
            "rel_tol": experiment.inputs.rel_tol,
            "dtype": args.dtype,
            "modules": args.modules,
        },
        config={"pencil": ctx.cfg["pencil"]},
        seed=args.seed,
        checkpoint=f"run{args.run}",
        out_dir=ctx.out("results"),
    )


def plot(path, results: dict, names: list[str]) -> None:
    """Grouped bars: each method's max |Δ log-prob| on the factor prompts, one bar per condition number."""
    fig, ax = subplots(figsize=(7, 3))
    bar_width = BAR_GROUP_WIDTH / len(results)
    for i, (cond, cond_results) in enumerate(results.items()):
        positions = [j + i * bar_width for j in range(len(names))]
        heights = [max(cond_results[name]["factor"]["max_abs_logprob_diff"], LOG_PLOT_FLOOR) for name in names]
        ax.bar(positions, heights, bar_width, label=f"cond {cond}")
    ax.set_yscale("log")
    group_centres = [j + bar_width * (len(results) - 1) / 2 for j in range(len(names))]
    tick_labels = [name.replace("_", "\n") for name in names]
    ax.set_xticks(group_centres, tick_labels, fontsize=7)
    ax.set_ylabel("max |Δ log-prob|, original vs\nreparameterised repair")
    ax.legend(fontsize=7)
    save(fig, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common_args(parser)
    parser.add_argument("--run", default="A")
    parser.add_argument("--conds", type=float, nargs="+", default=[10.0, 30.0])
    parser.add_argument("--mu", type=float, default=0.1, help="pencil / AlignMerge μ")
    parser.add_argument("--mu-regmean", type=float, default=3.0)
    parser.add_argument("--gamma-e", type=float, default=1e-2)
    parser.add_argument("--alignmerge-rank", type=int, default=64)
    parser.add_argument("--resta-alpha", type=float, default=0.25)
    parser.add_argument("--safelora-threshold", type=float, default=0.5)
    parser.add_argument("--modules", choices=["vo", "all"], default="vo")
    parser.add_argument(
        "--n-prompts", type=int, default=16, help="prompts per comparison set (log-prob tables live in RAM)"
    )
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64", "bfloat16"])
    parser.add_argument("--reestimate-layers", default=None, help="comma list (default: first two selection layers)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", nargs="+", default=None, help="default: every method")
    parser.add_argument("--out-name", default=None, help="result name (default invariance_run<run>)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    ctx = setup(args)
    experiment = prepare_experiment(ctx, args)
    results = {}
    for cond in args.conds:
        results[str(cond)] = compare_at_condition(ctx, args, experiment, cond)
    write_invariance_result(ctx, args, experiment, results)
    # the run is in the file name so a run on another setting doesn't overwrite the figure
    plot(ctx.out("figures") / f"{args.out_name or f'invariance_run{args.run}'}.png", results, list(experiment.methods))


if __name__ == "__main__":
    main()
