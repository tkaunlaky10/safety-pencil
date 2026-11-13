"""Repair, μ sweep, trust-region check and forecast (Prompt 6; predictions P2 and P3).

sweep     For one run, repair every module at each μ and measure on dev the safety drift S = KL(θ0 ‖ θ) on θ0's
          own answers to harmful prompts (per language and per prompt) and the task NLL. --method scaled (the
          config default since 13 Nov) is the scaled repair: one basis solve, the μ grid placed from the
          predicted regret before any measurement (pencil/mu_grid.py) and written to the result first, rule 11
          checked as "S fell and every trust ratio ≤ 1". --method covariant is the 12 Nov method: the rule-11
          trust region (covariant first, Euclidean fallback). Logs predicted S and R, step ratios (Euclidean and
          input-weighted) and, with --grad-diag, the task-gradient diagnostic. --save writes merged
          checkpoints for chosen μ to outputs/ckpt/repaired/ for 02_evaluate.py. --layers repairs only some
          layers (the row-set comparison on the selection layers).
forecast  For every adapter on disk (runs × fractions) and language: ½ Σ tr(G_h ΔW A_h ΔWᵀ), correlated with
          the measured refusal drop from 02_evaluate.py results named probe_theta0 and probe_run{R}_frac{F}.
          A ranking, not a number.
drift     For every adapter on disk and language: the measured drift S on dev (what ½ΔᵀF_hΔ forecasts) and
          the task NLL -> drift_S.json. Needs no factors; jobs/report.py pairs it with the forecast (P2).

    python scripts/06_repair.py sweep --run A                      # scaled repair, predicted μ grid
    python scripts/06_repair.py sweep --run A --method covariant
    python scripts/06_repair.py sweep --run A --layers selection --row-set all     # row-set comparison
    python scripts/06_repair.py sweep --run A --save 0.1 0.3
    python scripts/06_repair.py forecast
    python scripts/06_repair.py drift
    python scripts/06_repair.py sweep --run A --dry-run
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
from dataclasses import dataclass, field

import torch

from pencil.delta import as_dense, lora_delta, model_weights, save_checkpoint, set_weights
from pencil.modules import layer_order, parse_name
from pencil.mu_grid import mu_for_regret_fractions
from pencil.plots import save, subplots
from pencil.repair import (
    ModelRepair,
    TaskGradient,
    forecast_many,
    scaled_trust_region_repair,
    task_grad_diagnostic,
    task_gradient,
    trust_region_repair,
)
from pencil.results import write_result
from pencil.runtime import (
    DevProbe,
    RepairInputs,
    adapter_dir,
    available_langs,
    common_args,
    encode_pairs,
    harm_factors,
    load_model,
    load_repair_inputs,
    load_tokenizer,
    make_batches,
    prepare_factors,
    repair_kwargs,
    scaled_repair_kwargs,
    setup,
)
from pencil.stats import spearman_bootstrap_ci

REPAIRED_LABEL = "research artifact: pencil-repaired checkpoint, mu={mu}, from run {run}"
SCALED = ("scaled", None)  # SweepState.repairers key of the scaled repair
GRAD_DIAG_TASK_ROWS = 64  # task dev rows used by the task-gradient diagnostic
SAVE_MU_TOLERANCE = 1e-12  # a --save value matches a swept μ within this
MIN_FORECAST_PAIRS = 4  # fewer checkpoint-language pairs than this give no Spearman correlation
LOG_PLOT_FLOOR = 1e-12  # zero values are drawn here on log axes


# ---------------------------------------------------------------- sweep


def result_tag(run: str, frac: int, row_set, layers: str | None) -> str:
    """File-name tag for a sweep, e.g. 'runA_frac100_full' or 'runA_frac100_all_3-11'."""
    row_set_name = row_set if isinstance(row_set, str) else "mixed"
    tag = f"run{run}_frac{frac}_{row_set_name}"
    if layers:
        tag += "_" + layers.replace(",", "-")
    return tag


def modules_in_layers(ctx, deltas: dict, layers: str | None) -> list[str] | None:
    """The modules to repair, from --layers ('selection' or e.g. '3,11'); None means every module."""
    if not layers:
        return None
    if layers == "selection":
        keep = set(ctx.cfg["factors"]["selection_layers"])
    else:
        keep = {int(layer) for layer in layers.split(",")}
    return [module for module in deltas if parse_name(module)[1] in keep]


def fine_tuned_weights(base_weights: dict, deltas: dict) -> dict:
    """θ1 = W0 + ΔW for every module, added in float64 and stored in the base weights' dtype."""
    theta1 = {}
    for module, delta in deltas.items():
        W0 = base_weights[module]
        theta1[module] = (W0.double() + as_dense(delta, torch.float64)).to(W0.dtype)
    return theta1


def repair_kwargs_without_damping(pencil_cfg: dict, rel_tol: float, row_set) -> dict:
    """ModelRepair arguments from the config, minus damping and γ, which each repair setting chooses itself."""
    kwargs = repair_kwargs(pencil_cfg, rel_tol, row_set=row_set)
    del kwargs["damping"]
    del kwargs["gamma"]
    return kwargs


@dataclass
class SweepState:
    """What every μ of one sweep shares. `work` is the model that gets measured. θ1 is always the whole
    fine-tune, even when only some layers are repaired; all weight dicts stay on the CPU."""

    inputs: RepairInputs
    deltas: dict  # ΔW of the modules being repaired
    modules: list[str] | None  # the repaired modules; None means every module
    row_set: str | dict
    tag: str
    W0: dict  # θ0 weights of the repaired modules
    theta1: dict  # θ1 weights of every fine-tuned module
    work: torch.nn.Module
    probe: DevProbe
    drift: dict  # the dev probe on θ1
    nll_theta0: float
    repair_kw: dict  # ModelRepair arguments other than damping and γ
    repairers: dict  # one ModelRepair per (damping, γ), or SCALED, reused across μ
    method: str  # "scaled" or "covariant"
    scaled_kw: dict  # ModelRepair arguments of the scaled repair
    frontiers: dict = field(default_factory=dict)  # tuple(μ) -> predicted frontier (each costs a prediction per μ)
    task_gradient: TaskGradient | None = None  # g at θ1 for the scaled sweep's --grad-diag (μ-independent)

    def measured_S(self, new_weights: dict | None) -> float:
        """Mean dev drift with `new_weights` loaded into `work` (for the trust region); None means unrepaired."""
        if new_weights is None:
            return self.drift["S_mean"]
        set_weights(self.work, new_weights)
        return self.probe.measure(self.work)["S_mean"]

    def scaled_repairer(self) -> ModelRepair:
        """The scaled repair's ModelRepair, built on first use (its bases are solved once)."""
        if SCALED not in self.repairers:
            self.repairers[SCALED] = ModelRepair(self.inputs.harm_factors, self.inputs.task_factors, **self.scaled_kw)
        return self.repairers[SCALED]

    def predicted_frontier(self, pencil_cfg: dict, mus: list[float]) -> dict:
        """The reported repair's predicted frontier over `mus`, computed once per grid."""
        key = tuple(mus)
        if key not in self.frontiers:
            self.frontiers[key] = self.default_repairer(pencil_cfg).frontier(self.deltas, mus)
        return self.frontiers[key]

    def default_repairer(self, pencil_cfg: dict) -> ModelRepair:
        """The repair whose predicted frontier the result reports: the scaled one, or the configured damping."""
        if self.method == "scaled":
            return self.scaled_repairer()
        return self.repairer(pencil_cfg["damping"], pencil_cfg["gamma_c"])

    def repairer(self, damping: str, gamma: float) -> ModelRepair:
        """The ModelRepair for one damping setting, built on first use."""
        key = (damping, gamma)
        if key not in self.repairers:
            self.repairers[key] = ModelRepair(
                self.inputs.harm_factors, self.inputs.task_factors, damping=damping, gamma=gamma, **self.repair_kw
            )
        return self.repairers[key]


def prepare_sweep(ctx, args) -> SweepState:
    """Load θ0, the factors and the fine-tune, build θ1 on a working copy and measure its drift on dev."""
    pencil_cfg = ctx.cfg["pencil"]
    selection = args.layers == "selection"
    inputs = load_repair_inputs(ctx, args.run, args.frac, langs=args.langs, pool=args.pool, selection=selection)
    modules = modules_in_layers(ctx, inputs.deltas, args.layers)
    deltas = inputs.deltas
    if modules is not None:
        deltas = {module: inputs.deltas[module] for module in modules}
    row_set = args.row_set
    if not row_set:
        row_set = pencil_cfg.get("repair_row_set", "all")
    method = args.method or pencil_cfg.get("repair_method", "covariant")
    if method == "scaled" and modules is not None:
        raise SystemExit("--method scaled repairs the whole model; use --method covariant with --layers")

    full_delta = lora_delta(adapter_dir(ctx, args.run, args.frac), lowrank=True)
    W0_all = model_weights(inputs.model0, full_delta, device="cpu")
    theta1 = fine_tuned_weights(W0_all, full_delta)
    work = set_weights(copy.deepcopy(inputs.model0), theta1)
    probe = DevProbe(ctx, inputs.model0, inputs.tokenizer, inputs.langs)
    drift = probe.measure(work)
    nll_theta0 = probe.task_nll(inputs.model0)
    drift_by_lang = ", ".join(f"{lang} {S:.3g}" for lang, S in drift["S"].items())
    print(f"drift S {drift_by_lang}; task NLL θ0 {nll_theta0:.3f} θ1 {drift['task_nll']:.3f}")

    return SweepState(
        inputs=inputs,
        deltas=deltas,
        modules=modules,
        row_set=row_set,
        tag=result_tag(args.run, args.frac, row_set, args.layers),
        W0={module: W0_all[module] for module in deltas},
        theta1=theta1,
        work=work,
        probe=probe,
        drift=drift,
        nll_theta0=nll_theta0,
        repair_kw=repair_kwargs_without_damping(pencil_cfg, inputs.rel_tol, row_set),
        repairers={},
        method=method,
        scaled_kw=scaled_kw(pencil_cfg, inputs.rel_tol, row_set, method),
    )


def scaled_kw(pencil_cfg: dict, rel_tol: float, row_set, method: str) -> dict:
    """The scaled repair's ModelRepair arguments, or {} for the covariant method."""
    if method != "scaled":
        return {}
    return scaled_repair_kwargs(pencil_cfg, rel_tol, row_set=row_set)


def repair_at(state: SweepState, pencil_cfg: dict, mu: float) -> tuple[dict, dict, dict]:
    """Repair at one μ. The scaled repair is one attempt checked by rule 11; a covariant whole-model repair goes
    through the rule-11 trust region (covariant first, then Euclidean); a repair of some layers uses the
    configured damping only. Returns (weights, report, setting)."""
    if state.method == "scaled":
        return scaled_trust_region_repair(
            state.scaled_repairer(), state.W0, state.deltas, mu, state.measured_S, state.task_gradient
        )
    if state.modules is None:
        return trust_region_repair(
            state.inputs.harm_factors,
            state.inputs.task_factors,
            state.W0,
            state.deltas,
            mu,
            gamma_c=pencil_cfg["gamma_c"],
            gamma_e_grid=pencil_cfg["gamma_e_grid"],
            step_ratio_key=pencil_cfg.get("trust_region_step_ratio", "max_step_ratio"),
            actual_S=state.measured_S,
            repairers=state.repairers,
            **state.repair_kw,
        )
    damping = pencil_cfg["damping"]
    gamma = pencil_cfg["gamma_c"]
    new_weights, report = state.repairer(damping, gamma).repair(state.W0, state.deltas, mu)
    setting = {"damping": damping, "gamma": gamma, "attempts": []}
    return new_weights, report, setting


PER_MODULE_KEYS = {  # sweep point field -> per-module report key
    "step_ratios": "step_ratio",
    "input_step_ratios": "input_step_ratio",
    "step_ratios_unscaled": "step_ratio_unscaled",
    "task_input_step_ratios": "task_input_step_ratio",
    "task_input_step_ratios_unscaled": "task_input_step_ratio_unscaled",
    "trust_ratios": "trust_ratio",
    "scales": "scale",
    "harm_input_step_ratios": "harm_input_step_ratio",  # diagnostic only
}


def _median_ratio(report: dict, key: str) -> float:
    """The median over modules of one step ratio ("step_ratio" or "input_step_ratio")."""
    return statistics.median(stats[key] for stats in report["modules"].values())


def sweep_point(mu: float, setting: dict, measured: dict, report: dict, drift: dict) -> dict:
    """One μ's record: the damping used, measured S and task NLL, and the K-FAC predictions and step ratios."""
    per_module = report["modules"]
    return {
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
        "max_task_input_step_ratio": report["max_task_input_step_ratio"],
        "max_trust_ratio": report["max_trust_ratio"],
        "max_harm_input_step_ratio": report["max_harm_input_step_ratio"],
        "n_scaled": report["n_scaled"],
        "median_step_ratio": _median_ratio(report, "step_ratio"),
        "median_input_step_ratio": _median_ratio(report, "input_step_ratio"),
        "S_prompts": measured["S_prompts"],
        **per_module_fields(per_module),
    }


def per_module_fields(per_module: dict) -> dict:
    """{field: {module: value}} for every PER_MODULE_KEYS field."""
    fields = {}
    for name, key in PER_MODULE_KEYS.items():
        fields[name] = {module: stats[key] for module, stats in per_module.items()}
    return fields


def grad_diag_batches(ctx, state: SweepState) -> list[dict]:
    """The first GRAD_DIAG_TASK_ROWS task dev rows, one per batch, for the task-gradient diagnostic."""
    tokenizer = state.inputs.tokenizer
    pairs = [(row["prompt"], row["answer"]) for row in state.probe.task[:GRAD_DIAG_TASK_ROWS]]
    encoded = encode_pairs(tokenizer, pairs, ctx.cfg["factors"]["max_len"])
    return make_batches(encoded, tokenizer.pad_token_id, 1)


def theta1_task_gradient(ctx, state: SweepState) -> TaskGradient:
    """g at θ1 over the repaired modules, once per sweep (it does not depend on μ); `work` holds θ1 then."""
    set_weights(state.work, state.theta1)
    return task_gradient(state.work, grad_diag_batches(ctx, state), list(state.deltas))


def task_gradient_check(ctx, state: SweepState, setting: dict, mu: float, new_weights: dict, R_pred: float) -> dict:
    """The diagnostic |gᵀδ| / (½ δᵀĤδ), with the task gradient g taken at θ1 over the repaired modules.
    `work` is set to θ1 for it and holds the repaired weights again afterwards."""
    repairer = state.repairers[(setting["damping"], setting["gamma"])]
    steps = repairer.steps(state.deltas, mu, state.modules)
    set_weights(state.work, state.theta1)
    diagnostic = task_grad_diagnostic(state.work, grad_diag_batches(ctx, state), steps, R_pred)
    set_weights(state.work, new_weights)
    return diagnostic


def print_point(mu: float, setting: dict, measured: dict, report: dict, drift_S_mean: float) -> None:
    """One line per μ: measured vs predicted S, task NLL and the largest step ratios."""
    S_by_lang = ", ".join(f"{lang} {S:.4g}" for lang, S in measured["S"].items())
    message = (
        f"mu={mu:g} [{setting['damping']} {setting['gamma']:g}]: S {measured['S_mean']:.4g} ({S_by_lang}; "
        f"drift {drift_S_mean:.4g}, predicted {report['S_after_pred']:.3g}), task NLL {measured['task_nll']:.3f}, "
        f"step ratio max {report['max_step_ratio']:.2f} median {_median_ratio(report, 'step_ratio'):.2f}, "
        f"input-weighted max {report['max_input_step_ratio']:.2f} "
        f"median {_median_ratio(report, 'input_step_ratio'):.2f}, "
        f"trust ratio max {report['max_trust_ratio']:.3f} ({report['n_scaled']} modules scaled), "
        f"harmful-input max {report['max_harm_input_step_ratio']:.2f}"
    )
    if "task_grad_diagnostic" in report:
        message += f", |gᵀδ|/(½δᵀĤδ) {report['task_grad_diagnostic']['ratio']:.3g}"
    if setting.get("failed"):
        message += "  TRUST REGION FAILED: do not evaluate this point"
    print(message)


def save_requested(mu: float, save_mus: list[float] | None, save_all: bool = False) -> bool:
    """True with --save-all, or if μ is one of the --save values."""
    if save_all:
        return True
    if not save_mus:
        return False
    return any(abs(mu - save_mu) < SAVE_MU_TOLERANCE for save_mu in save_mus)


def sweep_one_mu(ctx, args, state: SweepState, mu: float) -> dict:
    """Repair at μ, measure on dev, print a line and save the checkpoint if asked. Returns the μ's record."""
    new_weights, report, setting = repair_at(state, ctx.cfg["pencil"], mu)
    # modules outside the repaired set keep θ1's weights
    set_weights(state.work, state.theta1)
    set_weights(state.work, new_weights)
    measured = state.probe.measure(state.work)
    point = sweep_point(mu, setting, measured, report, state.drift)
    if "task_grad_diagnostic" in report:  # the scaled repair computes it as it builds each step
        point["task_grad_diagnostic"] = report["task_grad_diagnostic"]
    elif args.grad_diag:
        point["task_grad_diagnostic"] = task_gradient_check(ctx, state, setting, mu, new_weights, report["R_pred"])
    print_point(mu, setting, measured, report, state.drift["S_mean"])
    if save_requested(mu, args.save, args.save_all):
        folder = ctx.out("ckpt", "repaired", f"{state.tag}_mu{mu:g}")
        path = save_checkpoint(state.work, state.inputs.tokenizer, folder, REPAIRED_LABEL.format(mu=mu, run=args.run))
        print(f"  saved {path}")
    return point


def spearman_pred_vs_actual(points: list[dict]) -> dict:
    """Spearman ρ with a bootstrap CI between predicted and measured S over the sweep, or the error message
    when there are too few points for a CI."""
    predicted = [point["S_pred"] for point in points]
    measured = [point["S_actual_mean"] for point in points]
    try:
        return spearman_bootstrap_ci(predicted, measured)
    except Exception as error:
        return {"error": str(error)}


def _repairers_by_name(state: SweepState) -> dict[str, ModelRepair]:
    """The sweep's ModelRepairs keyed "<damping>:<γ>", e.g. "covariant:0.001", and "scaled"."""
    named = {}
    for key, repairer in state.repairers.items():
        if key == SCALED:
            named["scaled"] = repairer
        else:
            damping, gamma = key
            named[f"{damping}:{gamma:g}"] = repairer
    return named


def write_sweep_result(ctx, args, state: SweepState, points: list[dict], mus: list[float], grid=None) -> None:
    """Write the sweep's result JSON (with the predicted frontier, the μ grid rule and the ceiling diagnostic of
    the reported repair) and, once a point is measured, its figure."""
    pencil_cfg = ctx.cfg["pencil"]
    default_repairer = state.default_repairer(pencil_cfg)
    spearman = spearman_pred_vs_actual(points)
    name = f"repair_sweep_{state.tag}"
    if args.out_name:
        name = args.out_name
    result = {
        "points": points,
        "S_drift": state.drift["S"],
        "task_nll": {"theta0": state.nll_theta0, "theta1": state.drift["task_nll"]},
        "method": state.method,
        "mu_grid_rule": grid,
        "ceiling_diagnostic": default_repairer.ceiling_report(),
        "frontier_pred": state.predicted_frontier(pencil_cfg, mus),
        "spearman_pred_vs_actual_S": spearman,
        "rel_tol": state.inputs.rel_tol,
        "null_floors": state.inputs.floors,
        "jitter": {name: repairer.jitters() for name, repairer in _repairers_by_name(state).items()},
        "langs": state.inputs.langs,
        "row_set": state.row_set,
        "modules": state.modules,
        "frac": args.frac,
    }
    path = write_result(
        name,
        result,
        config={"pencil": pencil_cfg},
        checkpoint=str(adapter_dir(ctx, args.run, args.frac)),
        out_dir=ctx.out("results"),
    )
    if points:
        plot_sweep(ctx.out("figures") / f"{name}.png", points)
    print(f"-> {path}")


def predicted_mu_grid(ctx, state: SweepState) -> dict:
    """The scaled repair's μ grid: predicted R(μ)/R(0) = each regret fraction (no forward passes)."""
    fractions = ctx.cfg["pencil"]["scaled"]["regret_fractions"]
    repairer = state.scaled_repairer()

    def predicted_R(mu: float) -> float:
        return repairer.predicted(state.deltas, mu)["R_pred"]

    grid = mu_for_regret_fractions(predicted_R, fractions)
    pairs = ", ".join(f"{fraction:g} -> {mu:.4g}" for fraction, mu in zip(fractions, grid["mu"]))
    print(f"μ grid from predicted R(μ)/R(0): {pairs}")
    ceiling = repairer.ceiling_report()
    print(f"ceiling diagnostic (report only): {len(ceiling['flagged'])} sides above 1/γ_c: {ceiling['flagged']}")
    return grid


def sweep(ctx, args) -> None:
    """The μ sweep for one run: drift before repair, then repair, measure and record every μ. The scaled repair
    without --mu writes its predicted μ grid to the result before measuring anything."""
    state = prepare_sweep(ctx, args)
    if args.grad_diag and state.method == "scaled":
        state.task_gradient = theta1_task_gradient(ctx, state)
    mus = args.mu
    grid = None
    if not mus and state.method == "scaled":
        grid = predicted_mu_grid(ctx, state)
        mus = grid["mu"]
        write_sweep_result(ctx, args, state, [], mus, grid)
    if not mus:
        mus = ctx.cfg["pencil"]["mu_grid"]
    points = []
    for mu in mus:
        points.append(sweep_one_mu(ctx, args, state, mu))
        # after every μ, so a later crash keeps the earlier ones
        write_sweep_result(ctx, args, state, points, mus, grid)


def plot_sweep(path, points: list[dict]) -> None:
    """Left: measured S against task NLL per language (the measured frontier). Right: predicted vs measured S."""
    fig, (frontier_ax, prediction_ax) = subplots(1, 2, figsize=(8, 3.2))
    task_nll = [point["task_nll"] for point in points]
    for lang in points[0]["S_actual"]:
        S_lang = [point["S_actual"][lang] for point in points]
        frontier_ax.plot(task_nll, S_lang, "o-", label=lang)
    frontier_ax.set(xlabel="task NLL on dev (lower is better)", ylabel="measured S = KL(θ0‖θ) on dev")
    frontier_ax.set_title("measured frontier", fontsize=9)
    frontier_ax.legend(fontsize=7)

    predicted = [max(point["S_pred"], LOG_PLOT_FLOOR) for point in points]
    measured = [max(point["S_actual_mean"], LOG_PLOT_FLOOR) for point in points]
    prediction_ax.loglog(predicted, measured, "o")
    prediction_ax.set(xlabel="predicted S (K-FAC, a ranking)", ylabel="measured S")
    prediction_ax.set_title("prediction vs measurement", fontsize=9)
    save(fig, path)


# ---------------------------------------------------------------- forecast


def adapter_deltas(ctx) -> dict[str, dict]:
    """The low-rank ΔW of every adapter on disk, keyed 'run|frac'; missing adapters are reported and skipped."""
    checkpoints = {}
    for run in ctx.cfg["runs"]:
        for fraction in ctx.cfg["save_fractions"]:
            frac = round(fraction * 100)
            path = adapter_dir(ctx, run, frac)
            if path.exists():
                checkpoints[f"{run}|{frac}"] = lora_delta(path, lowrank=True)
            else:
                print(f"missing adapter {path}: run 03_finetune.py --run {run} first")
    return checkpoints


def refusal_rates(results_dir, name):
    """{lang: refusal rate} from a 02_evaluate.py result, or None if it doesn't exist yet."""
    path = results_dir / f"{name}.json"
    if not path.exists():
        return None
    metrics = json.loads(path.read_text())["result"]["metrics"]
    return {lang: m["harmful"]["refusal"]["p"] for lang, m in metrics.items() if "harmful" in m}


def refusal_drop(rates_before: dict | None, rates_after: dict | None, lang: str) -> float | None:
    """Refusal rate at θ0 minus the rate after fine-tuning, or None if either is not measured for `lang`."""
    if not rates_before or not rates_after:
        return None
    if lang not in rates_before or lang not in rates_after:
        return None
    return rates_before[lang] - rates_after[lang]


def forecast_language(ctx, model0, lang: str, checkpoints: dict, row_set, rates_theta0: dict | None) -> list[dict]:
    """Forecast every checkpoint from one language's harmful factors and pair each with its refusal drop."""
    pencil_cfg = ctx.cfg["pencil"]
    # only the harmful side is used; it is passed as both sides to reuse the repair's factor preparation
    harm, _, _, _ = prepare_factors(model0, harm_factors(ctx, [lang]), harm_factors(ctx, [lang]), pencil_cfg)
    results_dir = ctx.out_root / "results"
    rows = []
    for key, per_module in forecast_many(harm, checkpoints, row_set).items():
        run, frac = key.split("|")
        rates_after = refusal_rates(results_dir, f"probe_run{run}_frac{frac}")
        drop = refusal_drop(rates_theta0, rates_after, lang)
        total = sum(per_module.values())
        rows.append(
            {
                "run": run,
                "frac": int(frac) / 100,
                "lang": lang,
                "forecast": total,
                "refusal_drop": drop,
                "per_module": {module: per_module[module] for module in layer_order(per_module)},
            }
        )
        message = f"{run} {int(frac) / 100:.2f} {lang}: forecast {total:.4g}"
        if drop is not None:
            message += f", refusal drop {drop:+.3f}"
        print(message)
    return rows


def forecast_spearman(rows: list[dict]) -> dict | None:
    """Spearman(forecast, refusal drop) over the rows that have a measured drop; None if there are too few."""
    measured_rows = [row for row in rows if row["refusal_drop"] is not None]
    if len(measured_rows) < MIN_FORECAST_PAIRS:
        return None
    forecasts = [row["forecast"] for row in measured_rows]
    drops = [row["refusal_drop"] for row in measured_rows]
    rho = spearman_bootstrap_ci(forecasts, drops)
    print(f"Spearman(forecast, refusal drop) over {len(measured_rows)} checkpoint-language pairs: {rho}")
    return rho


def forecast(ctx, args) -> None:
    """P2: one streaming pass over each language's harmful factors forecasts every checkpoint at once."""
    pencil_cfg = ctx.cfg["pencil"]
    model0 = load_model(ctx.aligned, ctx)
    langs = args.langs
    if not langs:
        langs = available_langs(ctx)
    row_set = pencil_cfg.get("forecast_row_set", "all")
    checkpoints = adapter_deltas(ctx)
    rates_theta0 = refusal_rates(ctx.out_root / "results", "probe_theta0")
    rows = []
    for lang in langs:
        rows.extend(forecast_language(ctx, model0, lang, checkpoints, row_set, rates_theta0))
    write_result(
        "forecast_p2",
        {"rows": rows, "spearman": forecast_spearman(rows), "row_set": row_set, "langs": langs},
        config={"pencil": pencil_cfg},
        out_dir=ctx.out("results"),
    )


def drift(ctx, args) -> None:
    """P2's measured side: the dev drift S per language and the task NLL of θ0 + ΔW for every adapter on disk,
    each loaded into one working copy of θ0 in turn."""
    langs = args.langs
    if not langs:
        langs = ctx.cfg["languages"]
    tokenizer = load_tokenizer(ctx)
    model0 = load_model(ctx.aligned, ctx)
    probe = DevProbe(ctx, model0, tokenizer, langs)
    work = copy.deepcopy(model0)
    rows = []
    for key, delta in adapter_deltas(ctx).items():
        run, frac = key.split("|")
        W0 = model_weights(model0, delta, device="cpu")
        set_weights(work, fine_tuned_weights(W0, delta))
        measured = probe.measure(work)
        set_weights(work, W0)  # back to θ0 before the next adapter
        for lang, S in measured["S"].items():
            rows.append({"run": run, "frac": int(frac) / 100, "lang": lang, "S": S, "task_nll": measured["task_nll"]})
        drift_by_lang = ", ".join(f"{lang} {S:.3g}" for lang, S in measured["S"].items())
        print(f"{run} {int(frac) / 100:.2f}: S {drift_by_lang}; task NLL {measured['task_nll']:.3f}")
    write_result("drift_S", {"rows": rows, "langs": langs}, out_dir=ctx.out("results"))


# ---------------------------------------------------------------- command line


def build_parser() -> argparse.ArgumentParser:
    """The two subcommands, sweep and forecast, with their flags."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="cmd", required=True)

    sweep_parser = common_args(commands.add_parser("sweep"))
    sweep_parser.add_argument("--run", default="A")
    sweep_parser.add_argument("--frac", type=int, default=100)
    sweep_parser.add_argument("--langs", nargs="+")
    sweep_parser.add_argument("--pool", choices=["equal", "tokens"], default="equal")
    sweep_parser.add_argument("--row-set", default=None)
    sweep_parser.add_argument("--layers", default=None, help="'selection' or e.g. '3,11': repair only these layers")
    sweep_parser.add_argument(
        "--method", choices=["scaled", "covariant"], default=None, help="default pencil.repair_method"
    )
    sweep_parser.add_argument(
        "--mu", type=float, nargs="+", help="default: the predicted grid (scaled) or pencil.mu_grid (covariant)"
    )
    sweep_parser.add_argument("--save", type=float, nargs="+", help="save merged checkpoints for these μ")
    sweep_parser.add_argument("--save-all", action="store_true", help="save every swept μ (e.g. a predicted grid)")
    sweep_parser.add_argument("--out-name", default=None, help="result name (default repair_sweep_<tag>)")
    sweep_parser.add_argument(
        "--grad-diag", action="store_true", help="task-gradient diagnostic (needs weight gradients)"
    )

    forecast_parser = common_args(commands.add_parser("forecast"))
    forecast_parser.add_argument("--langs", nargs="+")

    drift_parser = common_args(commands.add_parser("drift"))
    drift_parser.add_argument("--langs", nargs="+", help="default: languages in the config")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    ctx = setup(args)
    if args.cmd == "sweep":
        sweep(ctx, args)
    elif args.cmd == "forecast":
        forecast(ctx, args)
    else:
        drift(ctx, args)


if __name__ == "__main__":
    main()
