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

import torch

from pencil.datasets import load_split
from pencil.delta import model_weights, set_weights
from pencil.generate import prompt_ids
from pencil.leverage import leverage_table, random_fraction, refusal_cosine, top_fraction, top_output_direction
from pencil.modules import leaf, parse_name
from pencil.plots import save, subplots
from pencil.repair import ModelRepair
from pencil.results import write_result
from pencil.runtime import DevProbe, common_args, harmful_rows, load_repair_inputs, repair_kwargs, setup


@torch.no_grad()
def last_token_states(model, tok, prompts) -> torch.Tensor:
    """(n_layers + 1, n_prompts, hidden): the residual stream at each prompt's last template token."""
    dev = next(model.parameters()).device
    states = []
    for prompt in prompts:
        hidden = model(
            input_ids=torch.tensor([prompt_ids(tok, prompt)], device=dev), output_hidden_states=True
        ).hidden_states
        states.append(torch.stack([h[0, -1].double().cpu() for h in hidden]))
    return torch.stack(states, 1)


def refusal_directions(ctx, model, tok, langs, n: int) -> torch.Tensor:
    """Difference of means, harmful minus benign, per layer output: (n_layers + 1, hidden)."""
    harmful = [r["prompt"] for lang in langs for r in harmful_rows(ctx, lang, "dev", n)]
    benign = [r["prompt"] for lang in langs for r in load_split(ctx.data_dir, "xstest", "dev", lang)[:n]]
    return last_token_states(model, tok, harmful).mean(1) - last_token_states(model, tok, benign).mean(1)


def main(argv=None):
    p = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--run", default="A")
    p.add_argument("--mu", type=float, default=0.1)
    p.add_argument("--frac", type=float, default=0.1, help="fraction of modules in the targeted repair")
    p.add_argument("--n-random", type=int, default=5)
    p.add_argument("--n-direction", type=int, default=64, help="prompts per language for the refusal direction")
    p.add_argument("--damping", default=None, help="default pencil.damping; use the setting 06_repair.py chose")
    p.add_argument("--gamma", type=float, default=None, help="default pencil.gamma_c")
    args = p.parse_args(argv)
    ctx = setup(args)
    pc = ctx.cfg["pencil"]
    inputs = load_repair_inputs(ctx, args.run)
    model0, tok, deltas = inputs.model0, inputs.tok, inputs.deltas
    overrides = {k: v for k, v in (("damping", args.damping), ("gamma", args.gamma)) if v is not None}
    repairer = ModelRepair(inputs.fh, inputs.ft, **repair_kwargs(pc, inputs.rel_tol, **overrides))

    r_hat = refusal_directions(ctx, model0, tok, inputs.langs, args.n_direction)
    cosines = {
        n: refusal_cosine(top_output_direction(repairer.basis(n)), r_hat[parse_name(n)[1] + 1], seed=parse_name(n)[1])
        for n in deltas
        if leaf(n) in ("o_proj", "down_proj")
    }  # layer L writes hidden_states[L + 1]
    print(
        f"refusal-direction cosine: {sum(c['p_value'] < 0.05 for c in cosines.values())}/{len(cosines)} modules "
        f"with p < 0.05 (max |cos| {max(c['abs_cos'] for c in cosines.values()):.3f})"
    )

    table = leverage_table(repairer, deltas)
    top = top_fraction(table, args.frac)
    probe = DevProbe(ctx, model0, tok, inputs.langs)
    W0 = model_weights(model0, deltas, device="cpu")
    work = copy.deepcopy(model0)

    def repair_only(modules):
        new, report = repairer.repair(W0, deltas, args.mu, modules=modules)
        return {
            **probe.measure(set_weights(work, new)),
            "n_modules": len(report["modules"]),
            "max_step_ratio": report["max_step_ratio"],
        }

    targeted = {
        "none": repair_only([]),
        "full": repair_only(None),
        "top": repair_only(top),
        "random": [repair_only(random_fraction(table, args.frac, seed=s)) for s in range(args.n_random)],
    }
    for k in ("none", "full", "top"):
        print(
            f"{k:6s}: S {targeted[k]['S_mean']:.4g}, task NLL {targeted[k]['task_nll']:.3f} "
            f"({targeted[k]['n_modules']} modules)"
        )
    random_S = [r["S_mean"] for r in targeted["random"]]
    print(f"random {args.frac:.0%} (x{args.n_random}): S {min(random_S):.4g} .. {max(random_S):.4g}")
    write_result(
        f"localisation_run{args.run}",
        {"refusal_cosine": cosines, "targeted": targeted, "top_modules": top, "mu": args.mu, "frac": args.frac},
        config={"pencil": pc},
        checkpoint=f"run{args.run}",
        out_dir=ctx.out("results"),
    )
    plot(ctx.out("figures") / "localisation.png", cosines, targeted)


def plot(path, cosines: dict, targeted: dict):
    fig, ax = subplots(1, 2, figsize=(8, 3))
    for module, marker in (("o_proj", "o"), ("down_proj", "s")):
        points = sorted((parse_name(n)[1], c["abs_cos"]) for n, c in cosines.items() if leaf(n) == module)
        ax[0].plot(*zip(*points), marker + "-", label=module, ms=3)
    ax[0].plot(
        *zip(*sorted((parse_name(n)[1], c["null_q95"]) for n, c in cosines.items())), "k:", lw=1, label="random 95%"
    )
    ax[0].set(xlabel="layer", ylabel="|cos(top output direction, r̂)|")
    ax[0].legend(fontsize=7)
    labels = ["none", "full", "top"] + [f"rand{i}" for i in range(len(targeted["random"]))]
    values = [targeted[k]["S_mean"] for k in ("none", "full", "top")] + [r["S_mean"] for r in targeted["random"]]
    ax[1].bar(range(len(values)), values, color=["grey", "C0", "C1"] + ["C7"] * len(targeted["random"]))
    ax[1].set_xticks(range(len(values)), labels, fontsize=7, rotation=45)
    ax[1].set_ylabel("measured S on dev")
    ax[1].set_title("targeted (top 10% by leverage) vs random", fontsize=9)
    save(fig, path)


if __name__ == "__main__":
    main()
