"""Baselines on one run (Prompt 7), measured with the same dev probe as the pencil sweep (drift S per language
and task NLL). --save writes merged checkpoints for 02_evaluate.py. --finalists method=value runs only that
setting of each named method and saves it (the compare stage's test-tier finalists).

Ours (same factors as the pencil, so the comparison isolates the curvature model):
  regmean          W1 − ΔW A_h (A_h + μA_t)⁺                       μ in [0.3, 1000], 13 log-spaced
  diag_fisher      per-weight Fisher merge (per-sequence diagonal if the factor pass ran with --diag,
                   else diag(G) diag(A)ᵀ; printed)                 μ in [1e-3, 3], 11 log-spaced
  alignmerge       pencil with Euclidean rank-r harmful factors    r in {16, 64}, μ on the pencil grid
  fisher_standard  pencil with general-instruction factors at θ0 (needs factors/general/general)
  safelora         Safe LoRA projection C ΔW, C = VVᵀ/‖V‖, V = W_aligned − W_base
  safe_delta       SafeTune's Safe Delta fed our input factors
SafeTune's (on the merged drifted model; aligned and base models in bf16 on the CPU):
  wiseft, resta (DARE on), task_arithmetic, safemerge, lox
SafeTune's Pre/Post merge is not run: it computes (1 − α)·post + α·pre, which is WiSE-FT when pre is the
aligned model and pulls toward the unaligned model when pre is tiny-aya-base.

Every setting starts from θ1: the working copy is reset, the setting is applied, then measured and recorded.
A baseline that fails is recorded as an error row and the remaining baselines still run.

    python scripts/07_baselines.py --run A --methods wiseft resta safelora safemerge regmean diag_fisher
    python scripts/07_baselines.py --run A --dry-run
"""

from __future__ import annotations

import argparse
import copy
import math
from dataclasses import dataclass

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
NICE = ["alignmerge", "safe_delta", "task_arithmetic", "lox", "fisher_standard"]


def log_spaced(start: float, stop: float, count: int) -> list[float]:
    """`count` values from `start` to `stop`, evenly spaced in log10."""
    log_start = math.log10(start)
    log_range = math.log10(stop) - log_start
    # multiply before dividing: the grid values are recorded in results, so keep them bit-for-bit
    return [10 ** (log_start + i * log_range / (count - 1)) for i in range(count)]


REGMEAN_MUS = log_spaced(0.3, 1000, 13)
DIAG_FISHER_MUS = log_spaced(1e-3, 3, 11)
ALIGNMERGE_RANKS = (16, 64)
SAFELORA_THRESHOLDS = (0.3, 0.5, 0.7)
SAFE_DELTA_STRENGTHS = (0.05, 0.1, 0.2, 0.5)
FINALIST_TOLERANCE = 1e-9  # a --finalists value matches a grid value within this


@dataclass
class SafeTuneMethod:
    """A SafeTune recover trainer, the argument we sweep, its grid and the reference models it takes."""

    trainer: str
    swept_argument: str
    grid: list[float]
    references: set[str]


SAFETUNE = {
    "wiseft": SafeTuneMethod("WiseFTTrainer", "alpha", [0.25, 0.5, 0.75], {"aligned_model"}),
    "resta": SafeTuneMethod("ReStaTrainer", "alpha", [0.1, 0.25, 0.5], {"base_model", "aligned_model"}),
    "task_arithmetic": SafeTuneMethod(
        "TaskArithmeticTrainer", "alpha", [0.1, 0.25, 0.5], {"base_model", "aligned_model"}
    ),
    "safemerge": SafeTuneMethod("SafeMergeTrainer", "threshold", [0.25, 0.35, 0.5], {"base_model", "aligned_model"}),
    "lox": SafeTuneMethod("LoXTrainer", "extrapolation_factor", [0.1, 0.3, 0.5], {"base_model", "aligned_model"}),
}


def restrict_grid(values: list[float], wanted: float | None, method: str) -> list[float]:
    """The method's grid, or only its value matching `wanted` (a --finalists pick) when one is given."""
    if wanted is None:
        return list(values)
    matches = [value for value in values if math.isclose(value, wanted, rel_tol=FINALIST_TOLERANCE)]
    if not matches:
        raise ValueError(f"--finalists {method}={wanted!r} is not in its grid {values}")
    return matches[:1]


def save_tag(method: str, setting) -> str:
    """How --save names a setting, e.g. 'regmean:3' or 'alignmerge:r16_mu0.1'."""
    if isinstance(setting, float):
        return f"{method}:{setting:g}"
    return f"{method}:{setting}"


class BaselineRunner:
    """Holds θ0, a working copy, the factors and the dev probe. Each baseline calls `reset()` (work ← θ1),
    applies one setting to `work`, then `record(setting)`; rows collect per method."""

    def __init__(self, ctx, run: str, save_tags: list[str], finalists: dict[str, float] | None = None):
        self.ctx = ctx
        self.run = run
        self.save_tags = save_tags
        self.finalists = {}
        if finalists:
            self.finalists = finalists
        self.pencil_cfg = ctx.cfg["pencil"]
        self.inputs = load_repair_inputs(ctx, run)
        self.model0 = self.inputs.model0
        self.deltas = self.inputs.deltas
        self.work = copy.deepcopy(self.model0)
        self.row_set = single_row_set(self.pencil_cfg)
        self.probe = DevProbe(ctx, self.model0, self.inputs.tokenizer, self.inputs.langs)
        self._references = None
        self.method = ""
        self.rows: list[dict] = []
        self.setting_started = 0.0

    def reset(self) -> None:
        """work ← θ1: every parameter from θ0 (SafeTune methods may touch embeddings and norms), then + ΔW.
        Also starts the clock for the setting about to be applied."""
        with torch.no_grad():
            for param_work, param_theta0 in zip(self.work.parameters(), self.model0.parameters()):
                param_work.copy_(param_theta0)
        merge_deltas(self.work, self.deltas)
        self.setting_started = self.ctx.elapsed()

    def write_per_module(self, new_delta, value) -> None:
        """work ← W0 + new_delta(module, ΔW, value), one module at a time in layer order (ΔW in float64)."""
        with torch.no_grad():
            for module, delta in self.deltas.items():
                W0 = self.model0.get_submodule(module).weight
                dW_new = new_delta(module, as_dense(delta, torch.float64), value)
                W_new = W0.double().cpu() + dW_new.cpu()
                self.work.get_submodule(module).weight.copy_(W_new.to(W0))

    def references(self):
        """(aligned, base) models in bf16 on the CPU, loaded on first use."""
        if self._references is None:
            aligned = load_model(self.ctx.aligned, self.ctx, dtype=torch.bfloat16).cpu()
            base = load_model(self.ctx.base, self.ctx, dtype=torch.bfloat16).cpu()
            self._references = (aligned, base)
        return self._references

    def grid(self, values) -> list[float]:
        """The current method's grid values to run: all of them, or just its --finalists pick."""
        return restrict_grid(list(values), self.finalists.get(self.method), self.method)

    def start_method(self, method: str) -> None:
        """Begin collecting rows for a new baseline."""
        self.method = method
        self.rows = []

    def record(self, setting) -> None:
        """Measure `work` for one setting, add its row, print it, and save the checkpoint if --save asked."""
        measured = self.probe.measure(self.work)
        row = {"setting": setting, **measured, "seconds": self.ctx.elapsed() - self.setting_started}
        self.rows.append(row)
        print(f"{self.method} {setting}: S {row['S_mean']:.4g}, task NLL {row['task_nll']:.3f}")
        tag = save_tag(self.method, setting)
        if tag in self.save_tags:
            folder = self.ctx.out("ckpt", "baselines", f"run{self.run}_{tag.replace(':', '_')}")
            save_checkpoint(
                self.work, self.inputs.tokenizer, folder, f"research artifact: baseline {tag} on run {self.run}"
            )


# ---------------------------------------------------------------- per-module merges


@dataclass
class DiagonalFisherMerge:
    """Per-weight Fisher merge. Uses the per-sequence diagonal from the factor pass when both sides have one
    (`exact`), otherwise the K-FAC diagonal diag(G) diag(A)ᵀ."""

    harm: object
    task: object
    row_set: str
    exact: bool

    def diagonal(self, factors, module: str) -> torch.Tensor:
        if self.exact:
            return factors.get("D", module)
        return kfac_diagonal(*factors.pair(module, self.row_set))

    def delta(self, module: str, dW: torch.Tensor, mu: float) -> torch.Tensor:
        return diag_fisher_delta(dW, self.diagonal(self.harm, module), self.diagonal(self.task, module), mu)


@dataclass
class SafeLoRAProjection:
    """Safe LoRA: ΔW → C ΔW with C = V Vᵀ / ‖V‖, V = W_aligned − W_base, when cos(C ΔW, ΔW) < threshold."""

    aligned: torch.nn.Module
    base: torch.nn.Module

    def delta(self, module: str, dW: torch.Tensor, threshold: float) -> torch.Tensor:
        W_aligned = self.aligned.get_submodule(module).weight
        W_base = self.base.get_submodule(module).weight
        projected, _ = safe_lora_delta(dW, W_aligned, W_base, threshold)
        return projected


# ---------------------------------------------------------------- the baselines


def run_regmean(runner: BaselineRunner) -> None:
    """RegMean on our factors: W1 − ΔW A_h (A_h + μA_t)⁺ for each μ."""
    eig_device = repair_kwargs(runner.pencil_cfg, runner.inputs.rel_tol)["eig_device"]
    regmean = RegMean(
        runner.inputs.harm_factors,
        runner.inputs.task_factors,
        runner.row_set,
        top_k=runner.pencil_cfg.get("top_k_mlp"),
        eig_device=eig_device,
    )
    for mu in runner.grid(REGMEAN_MUS):
        runner.reset()
        runner.write_per_module(regmean.delta, mu)
        runner.record(mu)


def run_diag_fisher(runner: BaselineRunner) -> None:
    """Diagonal Fisher merge (Matena & Raffel) for each μ; prints which diagonal it uses."""
    harm = runner.inputs.harm_factors
    task = runner.inputs.task_factors
    exact = harm.has_diag and task.has_diag
    if exact:
        print("diag_fisher: per-sequence diagonal from the factor pass")
    else:
        print("diag_fisher: diag(G) diag(A)ᵀ fallback")
    merge = DiagonalFisherMerge(harm, task, runner.row_set, exact)
    for mu in runner.grid(DIAG_FISHER_MUS):
        runner.reset()
        runner.write_per_module(merge.delta, mu)
        runner.record(mu)


def run_pencil_grid(runner: BaselineRunner, harm, task, rel_tol: float, label_prefix: str = "") -> None:
    """The pencil repair with the given factors at every μ of the pencil grid. Settings are labelled by μ, or
    '<label_prefix><μ>' when a prefix is given."""
    repairer = ModelRepair(harm, task, **repair_kwargs(runner.pencil_cfg, rel_tol))
    W0 = model_weights(runner.model0, runner.deltas, device="cpu")
    for mu in runner.pencil_cfg["mu_grid"]:
        label = mu
        if label_prefix:
            label = f"{label_prefix}{mu:g}"
        runner.reset()
        repaired, _ = repairer.repair(W0, runner.deltas, mu)
        set_weights(runner.work, repaired)
        runner.record(label)


def run_alignmerge(runner: BaselineRunner) -> None:
    """AlignMerge-style: the pencil with Euclidean rank-r harmful factors, one rank after the other (so only
    one rank's bases are in memory)."""
    for rank in ALIGNMERGE_RANKS:
        harm = runner.inputs.harm_factors.with_transform(alignmerge_transform(rank))
        run_pencil_grid(runner, harm, runner.inputs.task_factors, runner.inputs.rel_tol, label_prefix=f"r{rank}_mu")


def run_fisher_standard(runner: BaselineRunner) -> None:
    """A standard Fisher merge: the pencil with general-instruction factors at θ0 in place of harmful ones."""
    general = LazyFactors([load_factors(runner.ctx, "general", "general")])
    task = task_factors(runner.ctx, runner.run)
    general_prepared, task_prepared, rel_tol, _ = prepare_factors(runner.model0, general, task, runner.pencil_cfg)
    run_pencil_grid(runner, general_prepared, task_prepared, rel_tol)


def run_safelora(runner: BaselineRunner) -> None:
    """Safe LoRA projection for each similarity threshold."""
    aligned, base = runner.references()
    projection = SafeLoRAProjection(aligned, base)
    for threshold in SAFELORA_THRESHOLDS:
        runner.reset()
        runner.write_per_module(projection.delta, threshold)
        runner.record(threshold)


def run_safe_delta(runner: BaselineRunner) -> None:
    """SafeTune's Safe Delta with our input factors as its Hessians, for each strength."""
    import safetune.recover as recover  # optional dependency, imported only when this baseline runs

    aligned, _ = runner.references()
    print("safe_delta: W_sft comes from bf16 merged weights (SafeTune's interface), unlike the other methods")
    hessians = safe_delta_hessians(
        runner.inputs.harm_factors, runner.row_set
    )  # every module's input factor (~19 GB fp32)
    for strength in SAFE_DELTA_STRENGTHS:
        runner.reset()
        recover.apply_safe_delta(
            runner.work, aligned=aligned, strength=strength, hessian_inputs=hessians, param_filter=["layers."]
        )
        runner.record(strength)


def run_safetune_trainer(runner: BaselineRunner, method: str) -> None:
    """One of SafeTune's recover trainers (table SAFETUNE) on the merged drifted model, over its grid."""
    import safetune.runner.recover as recover  # optional dependency, imported only when this baseline runs

    aligned, base = runner.references()
    spec = SAFETUNE[method]
    available = {"aligned_model": aligned, "base_model": base}
    trainer_kwargs = {name: available[name] for name in spec.references}
    if method == "resta":
        trainer_kwargs["dare"] = True
    trainer_class = getattr(recover, spec.trainer)
    for value in runner.grid(spec.grid):
        runner.reset()
        trainer_class(runner.work, **trainer_kwargs, **{spec.swept_argument: value}).apply()
        runner.record(value)


OUR_BASELINES = {
    "regmean": run_regmean,
    "diag_fisher": run_diag_fisher,
    "alignmerge": run_alignmerge,
    "fisher_standard": run_fisher_standard,
    "safelora": run_safelora,
    "safe_delta": run_safe_delta,
}


def run_method(runner: BaselineRunner, method: str) -> list[dict]:
    """Every setting of one baseline. A failure ends that baseline with an error row but doesn't stop the run."""
    runner.start_method(method)
    try:
        if method in OUR_BASELINES:
            OUR_BASELINES[method](runner)
        else:
            run_safetune_trainer(runner, method)
    except Exception as error:  # one broken baseline must not stop the rest
        message = f"{type(error).__name__}: {error}"
        runner.rows.append({"error": message})
        print(f"{method}: FAILED {message}")
    return runner.rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common_args(parser)
    parser.add_argument("--run", default="A")
    parser.add_argument("--methods", nargs="+", default=MUST, help=f"must: {MUST}; nice: {NICE}")
    parser.add_argument("--save", nargs="+", default=[], help="settings to save, e.g. regmean:3 wiseft:0.5")
    parser.add_argument("--out-name", default=None, help="result file name (default baselines_run<run>)")
    parser.add_argument(
        "--finalists",
        nargs="+",
        default=[],
        help="method=value: run only that setting of each method and save it (replaces --methods)",
    )
    return parser


def parse_finalists(pairs: list[str]) -> dict[str, float]:
    """{"regmean": 3.16, ...} from ["regmean=3.16", ...]."""
    finalists = {}
    for pair in pairs:
        method, separator, value = pair.partition("=")
        if not separator:
            raise SystemExit(f"--finalists takes method=value, got {pair!r}")
        finalists[method] = float(value)
    return finalists


def measure_unrepaired(runner: BaselineRunner) -> dict:
    """The dev probe on θ1 (the drift every baseline starts from) and the task NLL of θ0."""
    runner.reset()
    drift = runner.probe.measure(runner.work)
    task_nll_theta0 = runner.probe.task_nll(runner.model0)
    print(f"drift: S {drift['S']}, task NLL {drift['task_nll']:.3f}")
    return {"drift": drift, "theta0": {"task_nll": task_nll_theta0}}


def main(argv=None):
    args = build_parser().parse_args(argv)
    ctx = setup(args)
    finalists = parse_finalists(args.finalists)
    methods = args.methods
    save_tags = list(args.save)
    if finalists:
        methods = list(finalists)
        save_tags += [save_tag(method, value) for method, value in finalists.items()]
    runner = BaselineRunner(ctx, args.run, save_tags, finalists)
    results = measure_unrepaired(runner)
    for method in methods:
        results[method] = run_method(runner, method)

    out_name = args.out_name
    if out_name is None:
        out_name = f"baselines_run{args.run}"
    write_result(
        out_name,
        results,
        config={"pencil": runner.pencil_cfg, "methods": methods, "finalists": finalists},
        checkpoint=f"run{args.run}",
        out_dir=ctx.out("results"),
    )


if __name__ == "__main__":
    main()
