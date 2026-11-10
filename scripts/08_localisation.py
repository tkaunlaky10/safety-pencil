"""Localisation (Prompt 8 step 4; prediction P5).

1. Refusal direction per layer: mean residual stream at the last prompt token for harmful dev prompts minus
   that for benign XSTest dev prompts, pooled over languages (what SafeTune's RefusalDirection extracts,
   without keeping full-sequence activations for every layer).
2. For o_proj and down_proj, which write into the residual stream: |cos(top pencil output direction, r̂)|
   against random unit vectors.
3. Targeted repair: the top 10% of modules by leverage vs the full repair vs 5 random 10% draws (the
   random-module control), measured with the dev probe at one μ.

    python scripts/08_localisation.py --run A --mu 0.1
    python scripts/08_localisation.py --run A --dry-run
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass

import torch

from pencil.datasets import load_split
from pencil.delta import model_weights, set_weights
from pencil.generate import prompt_ids
from pencil.leverage import leverage_table, random_fraction, refusal_cosine, top_fraction, top_output_direction
from pencil.modules import RESIDUAL_WRITERS, leaf, parse_name
from pencil.plots import save, subplots
from pencil.repair import ModelRepair
from pencil.results import write_result
from pencil.runtime import DevProbe, common_args, harmful_rows, load_repair_inputs, repair_kwargs, setup

SIGNIFICANCE = 0.05
FIXED_REPAIRS = ("none", "full", "top")


# ---------------------------------------------------------------- refusal direction


@torch.no_grad()
def last_token_states(model, tokenizer, prompts) -> torch.Tensor:
    """(n_layers + 1, n_prompts, hidden): the residual stream at each prompt's last template token."""
    device = next(model.parameters()).device
    states = []
    for prompt in prompts:
        input_ids = torch.tensor([prompt_ids(tokenizer, prompt)], device=device)
        hidden_states = model(input_ids=input_ids, output_hidden_states=True).hidden_states
        last_token = [layer_states[0, -1].double().cpu() for layer_states in hidden_states]
        states.append(torch.stack(last_token))
    return torch.stack(states, 1)


def refusal_directions(ctx, model, tokenizer, langs, n_per_lang: int) -> torch.Tensor:
    """Difference of means, harmful minus benign, per layer output: (n_layers + 1, hidden)."""
    harmful = []
    benign = []
    for lang in langs:
        harmful.extend(row["prompt"] for row in harmful_rows(ctx, lang, "dev", n_per_lang))
        benign.extend(row["prompt"] for row in load_split(ctx.data_dir, "xstest", "dev", lang)[:n_per_lang])
    harmful_mean = last_token_states(model, tokenizer, harmful).mean(1)
    benign_mean = last_token_states(model, tokenizer, benign).mean(1)
    return harmful_mean - benign_mean


def refusal_cosines(repairer: ModelRepair, deltas: dict, r_hat: torch.Tensor) -> dict[str, dict]:
    """For each o_proj and down_proj: |cos(top pencil output direction, r̂)| and its random-vector null."""
    cosines = {}
    for module in deltas:
        if leaf(module) not in RESIDUAL_WRITERS:
            continue
        layer = parse_name(module)[1]
        direction = top_output_direction(repairer.basis(module))
        # layer L writes hidden_states[L + 1]; hidden_states[0] is the embedding output
        cosines[module] = refusal_cosine(direction, r_hat[layer + 1], seed=layer)
    return cosines


def print_cosine_summary(cosines: dict) -> None:
    significant = [c for c in cosines.values() if c["p_value"] < SIGNIFICANCE]
    largest = max(c["abs_cos"] for c in cosines.values())
    print(
        f"refusal-direction cosine: {len(significant)}/{len(cosines)} modules "
        f"with p < {SIGNIFICANCE} (max |cos| {largest:.3f})"
    )


# ---------------------------------------------------------------- targeted repair


def build_repairer(ctx, args, inputs) -> ModelRepair:
    """The pencil repair with the config's settings, or --damping / --gamma where given."""
    overrides = {}
    if args.damping is not None:
        overrides["damping"] = args.damping
    if args.gamma is not None:
        overrides["gamma"] = args.gamma
    return ModelRepair(
        inputs.harm_factors, inputs.task_factors, **repair_kwargs(ctx.cfg["pencil"], inputs.rel_tol, **overrides)
    )


@dataclass
class TargetedRepair:
    """Repairs only some modules at one μ (the others get W0 + ΔW) and measures the result with the dev probe."""

    repairer: ModelRepair
    probe: DevProbe
    work: torch.nn.Module
    W0: dict
    deltas: dict
    mu: float

    def measure(self, modules: list[str] | None) -> dict:
        """The dev probe, the number of repaired modules and the largest step ratio; None repairs every module."""
        new_weights, report = self.repairer.repair(self.W0, self.deltas, self.mu, modules=modules)
        set_weights(self.work, new_weights)
        measured = self.probe.measure(self.work)
        return {**measured, "n_modules": len(report["modules"]), "max_step_ratio": report["max_step_ratio"]}


def targeted_repairs(ctx, args, inputs, repairer: ModelRepair, table: dict, top: list[str]) -> dict:
    """No repair, the full repair, the top-leverage modules only, and --n-random random draws of as many."""
    probe = DevProbe(ctx, inputs.model0, inputs.tokenizer, inputs.langs)
    W0 = model_weights(inputs.model0, inputs.deltas, device="cpu")
    work = copy.deepcopy(inputs.model0)
    targeted = TargetedRepair(repairer, probe, work, W0, inputs.deltas, args.mu)

    results = {
        "none": targeted.measure([]),
        "full": targeted.measure(None),
        "top": targeted.measure(top),
    }
    random_draws = []
    for seed in range(args.n_random):
        random_modules = random_fraction(table, args.frac, seed=seed)
        random_draws.append(targeted.measure(random_modules))
    results["random"] = random_draws
    return results


def print_targeted_summary(targeted: dict, frac: float, n_random: int) -> None:
    for name in FIXED_REPAIRS:
        result = targeted[name]
        print(f"{name:6s}: S {result['S_mean']:.4g}, task NLL {result['task_nll']:.3f} ({result['n_modules']} modules)")
    random_S = [draw["S_mean"] for draw in targeted["random"]]
    print(f"random {frac:.0%} (x{n_random}): S {min(random_S):.4g} .. {max(random_S):.4g}")


# ---------------------------------------------------------------- output


def split_points(points: list[tuple[int, float]]) -> tuple[list[int], list[float]]:
    """(layers, values) of (layer, value) points sorted by layer, ready to plot as a line."""
    ordered = sorted(points)
    layers = [layer for layer, _ in ordered]
    values = [value for _, value in ordered]
    return layers, values


def plot(path, cosines: dict, targeted: dict, frac: float) -> None:
    """Left: |cos(top output direction, r̂)| per layer for o_proj and down_proj, with the random-vector 95%
    quantile. Right: measured S for no, full, top-leverage and random targeted repairs."""
    fig, (cosine_ax, repair_ax) = subplots(1, 2, figsize=(8, 3))
    for module_leaf, marker in (("o_proj", "o"), ("down_proj", "s")):
        points = []
        for module, cosine in cosines.items():
            if leaf(module) == module_leaf:
                points.append((parse_name(module)[1], cosine["abs_cos"]))
        layers, abs_cos = split_points(points)
        cosine_ax.plot(layers, abs_cos, marker + "-", label=module_leaf, ms=3)
    null_points = [(parse_name(module)[1], cosine["null_q95"]) for module, cosine in cosines.items()]
    null_layers, null_q95 = split_points(null_points)
    cosine_ax.plot(null_layers, null_q95, "k:", lw=1, label="random 95%")
    cosine_ax.set(xlabel="layer", ylabel="|cos(top output direction, r̂)|")
    cosine_ax.legend(fontsize=7)

    labels = list(FIXED_REPAIRS)
    values = [targeted[name]["S_mean"] for name in FIXED_REPAIRS]
    colors = ["grey", "C0", "C1"]
    for i, draw in enumerate(targeted["random"]):
        labels.append(f"rand{i}")
        values.append(draw["S_mean"])
        colors.append("C7")
    positions = range(len(values))
    repair_ax.bar(positions, values, color=colors)
    repair_ax.set_xticks(positions, labels, fontsize=7, rotation=45)
    repair_ax.set_ylabel("measured S on dev")
    repair_ax.set_title(f"targeted (top {frac:.0%} by leverage) vs random", fontsize=9)
    save(fig, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common_args(parser)
    parser.add_argument("--run", default="A")
    parser.add_argument("--mu", type=float, default=0.1)
    parser.add_argument("--frac", type=float, default=0.1, help="fraction of modules in the targeted repair")
    parser.add_argument("--n-random", type=int, default=5)
    parser.add_argument("--n-direction", type=int, default=64, help="prompts per language for the refusal direction")
    parser.add_argument("--damping", default=None, help="default pencil.damping; use the setting 06_repair.py chose")
    parser.add_argument("--gamma", type=float, default=None, help="default pencil.gamma_c")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    ctx = setup(args)
    inputs = load_repair_inputs(ctx, args.run)
    repairer = build_repairer(ctx, args, inputs)

    r_hat = refusal_directions(ctx, inputs.model0, inputs.tokenizer, inputs.langs, args.n_direction)
    cosines = refusal_cosines(repairer, inputs.deltas, r_hat)
    print_cosine_summary(cosines)

    table = leverage_table(repairer, inputs.deltas)
    top = top_fraction(table, args.frac)
    targeted = targeted_repairs(ctx, args, inputs, repairer, table, top)
    print_targeted_summary(targeted, args.frac, args.n_random)

    write_result(
        f"localisation_run{args.run}",
        {"refusal_cosine": cosines, "targeted": targeted, "top_modules": top, "mu": args.mu, "frac": args.frac},
        config={"pencil": ctx.cfg["pencil"]},
        checkpoint=f"run{args.run}",
        out_dir=ctx.out("results"),
    )
    # the run is in the file name so a run on another setting doesn't overwrite the figure
    plot(ctx.out("figures") / f"localisation_run{args.run}.png", cosines, targeted, args.frac)


if __name__ == "__main__":
    main()
