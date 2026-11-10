"""Localisation (Prompt 8 step 4; prediction P5).

1. Refusal direction per layer: difference of means of the residual stream at the last prompt position
   (the generation prompt), harmful dev prompts minus benign XSTest dev prompts, pooled over languages.
   This is what SafeTune's RefusalDirection extracts; computing it here avoids keeping full-sequence
   activations for every layer.
2. For o_proj and down_proj (they write into the residual stream): |cos(top pencil output direction, r̂_l)|
   with a null distribution of random unit vectors.
3. Targeted repair: the top 10% of modules by leverage vs the full repair vs 5 random 10% draws (the
   random-module control), measured with the dev probe (S and task NLL) at one μ.

    python scripts/08_localisation.py --run A --mu 0.1
    python scripts/08_localisation.py --run A --dry-run
"""
from __future__ import annotations

import argparse
import copy

import torch

from pencil.delta import lora_delta, set_weights
from pencil.evaluate import task_nll
from pencil.factors import parse_name
from pencil.generate import prompt_ids
from pencil.leverage import leverage_table, random_fraction, refusal_cosine, top_fraction, top_output_direction
from pencil.repair import ModelRepair, model_weights
from pencil.results import write_result
from pencil.runtime import (available_langs, common_args, dev_harm_batches, harm_factors, harmful_rows, load_model,
                            load_tokenizer, measure_S, model_shape, prepare_factors, repair_kwargs, setup,
                            task_dev_rows, task_factors)
from pencil.datasets import load_split


@torch.no_grad()
def last_token_states(model, tok, prompts, device):
    """(n_layers + 1, n, hidden) residual stream at the generation prompt's last token."""
    out = []
    for p in prompts:
        ids = torch.tensor([prompt_ids(tok, p)], device=device)
        hs = model(input_ids=ids, output_hidden_states=True).hidden_states
        out.append(torch.stack([h[0, -1].double().cpu() for h in hs]))
    return torch.stack(out, 1)


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
    tok = load_tokenizer(ctx)
    model0 = load_model(ctx.aligned, ctx)
    shape = model_shape(model0)
    langs = available_langs(ctx)

    # 1. refusal direction per layer
    harm = [r["prompt"] for l in langs for r in harmful_rows(ctx, l, "dev", args.n_direction)]
    benign = [r["prompt"] for l in langs for r in load_split(ctx.data_dir, "xstest", "dev", l)[:args.n_direction]]
    Hs, Bs = last_token_states(model0, tok, harm, ctx.device), last_token_states(model0, tok, benign, ctx.device)
    r_hat = Hs.mean(1) - Bs.mean(1)                                   # (n_layers + 1, hidden)

    # 2. cosine with the top pencil output direction (o_proj, down_proj write into the residual after layer l)
    deltas = lora_delta(ctx.out_root / "ckpt" / args.run / "frac100", lowrank=True)
    fh, ft, tol, _ = prepare_factors(model0, harm_factors(ctx, langs), task_factors(ctx, args.run), pc)
    deltas = {n: d for n, d in deltas.items() if n in fh.a_key}
    over = {k: v for k, v in (("damping", args.damping), ("gamma", args.gamma)) if v is not None}
    rep = ModelRepair(fh, ft, **repair_kwargs(pc, tol, **over))
    cos = {}
    for n in ModelRepair.order(deltas):
        _, l, leaf = parse_name(n)
        if leaf in ("o_proj", "down_proj"):
            cos[n] = refusal_cosine(top_output_direction(rep.basis(n)), r_hat[l + 1], seed=l)
    sig = sum(c["p_value"] < 0.05 for c in cos.values())
    print(f"refusal-direction cosine: {sig}/{len(cos)} modules with p < 0.05 "
          f"(max |cos| {max(c['abs_cos'] for c in cos.values()):.3f})")

    # 3. targeted vs full vs random repair
    table = leverage_table(rep, sorted(deltas))
    top = top_fraction(table, args.frac)
    dev_h = dev_harm_batches(ctx, model0, tok, langs, ctx.cfg["evaluation"]["probe_per_lang"])
    task_dev = task_dev_rows(ctx)
    W0 = model_weights(model0, deltas, device="cpu")
    work = copy.deepcopy(model0)

    def probe(mods):
        new, report = rep.repair(W0, deltas, args.mu, modules=mods)
        set_weights(work, new)
        S = measure_S(model0, work, dev_h)
        return {"S": S, "S_mean": sum(S.values()) / len(S), "task_nll": task_nll(work, tok, task_dev)["nll_per_token"],
                "n_modules": len(report["modules"]), "max_step_ratio": report["max_step_ratio"]}

    out = {"none": probe([]), "full": probe(None), "top": probe(top)}
    out["random"] = [probe(random_fraction(table, args.frac, seed=s)) for s in range(args.n_random)]
    for k in ("none", "full", "top"):
        print(f"{k:6s}: S {out[k]['S_mean']:.4g}, task NLL {out[k]['task_nll']:.3f} ({out[k]['n_modules']} modules)")
    rnd = [r["S_mean"] for r in out["random"]]
    print(f"random {args.frac:.0%} (x{args.n_random}): S {min(rnd):.4g} .. {max(rnd):.4g}")
    write_result(f"localisation_run{args.run}", {"refusal_cosine": cos, "targeted": out, "top_modules": top,
                                                  "mu": args.mu, "frac": args.frac},
                 config={"pencil": pc}, checkpoint=f"run{args.run}", out_dir=ctx.out("results"))
    plot(ctx, cos, out, shape["n_layers"])


def plot(ctx, cos, out, n_layers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    for leaf, mk in (("o_proj", "o"), ("down_proj", "s")):
        pts = sorted((parse_name(n)[1], c["abs_cos"], c["null_q95"]) for n, c in cos.items() if n.endswith(leaf))
        ax[0].plot([p[0] for p in pts], [p[1] for p in pts], mk + "-", label=leaf, ms=3)
    q = sorted((parse_name(n)[1], c["null_q95"]) for n, c in cos.items())
    ax[0].plot([p[0] for p in q], [p[1] for p in q], "k:", lw=1, label="random 95%")
    ax[0].set_xlabel("layer")
    ax[0].set_ylabel("|cos(top output direction, r̂)|")
    ax[0].legend(fontsize=7)
    labels = ["none", "full", "top"] + [f"rand{i}" for i in range(len(out["random"]))]
    vals = [out["none"]["S_mean"], out["full"]["S_mean"], out["top"]["S_mean"]] + [r["S_mean"] for r in out["random"]]
    ax[1].bar(range(len(vals)), vals, color=["grey", "C0", "C1"] + ["C7"] * len(out["random"]))
    ax[1].set_xticks(range(len(vals)), labels, fontsize=7, rotation=45)
    ax[1].set_ylabel("measured S on dev")
    ax[1].set_title("targeted (top 10% by leverage) vs random", fontsize=9)
    fig.tight_layout()
    fig.savefig(ctx.out("figures") / "localisation.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
