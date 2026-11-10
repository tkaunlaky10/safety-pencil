"""Leverage map: per layer and module, max α_i γ_j and the top-10 sum; per o_proj head and per v_proj KV group
lower bounds (Prompt 4 steps 3-4). Uses the same bases, row set and thresholds as the repair.

Writes outputs/results/leverage_{lang}_run{run}.json and outputs/figures/leverage_{lang}.png (and
leverage_heads_{lang}.png). Languages: each one with harmful factors on disk, plus "pooled" (equal weights).

    python scripts/04_leverage.py --run A
    python scripts/04_leverage.py --run A --dry-run
"""
from __future__ import annotations

import argparse
import math

from pencil.leverage import leverage_table
from pencil.repair import ModelRepair
from pencil.results import write_result
from pencil.runtime import (available_langs, common_args, harm_factors, load_model, model_shape, prepare_factors,
                            repair_kwargs, setup, task_factors)

MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def heatmap(table: dict, n_layers: int, path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    M = np.full((n_layers, len(MODULES)), np.nan)
    for row in table.values():
        if row["max"] > 0:
            M[row["layer"], MODULES.index(row["module"])] = math.log10(row["max"])
    fig, ax = plt.subplots(figsize=(4.8, 0.18 * n_layers + 1.4))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(MODULES)), [m.replace("_proj", "") for m in MODULES])
    ax.set_ylabel("layer")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, label="log10 max αγ")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def head_heatmap(table: dict, n_layers: int, n_heads: int, path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    M = np.full((n_layers, n_heads), np.nan)
    for row in table.values():
        if row["module"] == "o_proj" and "per_head_lower_bound" in row:
            M[row["layer"]] = [math.log10(v) if v > 0 else np.nan for v in row["per_head_lower_bound"]]
    fig, ax = plt.subplots(figsize=(0.3 * n_heads + 2, 0.18 * n_layers + 1.4))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xlabel("query head")
    ax.set_ylabel("layer")
    ax.set_title(title + "\n(per-head values are lower bounds on the module's top value)", fontsize=8)
    fig.colorbar(im, ax=ax, label="log10 αγ (lower bound)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    p = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--run", default="A")
    p.add_argument("--row-set", default=None, help="default pencil.repair_row_set")
    args = p.parse_args(argv)
    ctx = setup(args)
    pc = ctx.cfg["pencil"]
    model = load_model(ctx.aligned, ctx)            # only for shapes and LayerNorm weights
    shape = model_shape(model)
    ft0 = task_factors(ctx, args.run)
    langs = available_langs(ctx)
    targets = [(l, [l]) for l in langs] + ([("pooled", langs)] if len(langs) > 1 else [])
    row_set = args.row_set or pc.get("repair_row_set", "all")
    for name, ls in targets:
        fh, ft, tol, floors = prepare_factors(model, harm_factors(ctx, ls), ft0, pc)
        rep = ModelRepair(fh, ft, **repair_kwargs(pc, tol, row_set=row_set))
        mods = ModelRepair.order(m for m in fh.modules if m in ft.a_key)
        table = leverage_table(rep, mods, n_heads=shape["n_heads"], n_kv=shape["n_kv"], head_dim=shape["head_dim"])
        figs = ctx.out("figures")
        heatmap(table, shape["n_layers"], figs / f"leverage_{name}.png", f"leverage, harmful={name}, task=run {args.run}")
        head_heatmap(table, shape["n_layers"], shape["n_heads"], figs / f"leverage_heads_{name}.png",
                     f"o_proj per-head leverage, {name}")
        top = sorted(table, key=lambda n: -table[n]["max"])[:10]
        write_result(f"leverage_{name}_run{args.run}",
                     {"table": table, "top10_modules": top, "rel_tol": tol, "null_floors": floors, "row_set": row_set},
                     config={"pencil": pc, "langs": ls}, checkpoint=f"run{args.run}", out_dir=ctx.out("results"))
        print(f"{name}: top modules {[(t.split('layers.')[1], round(table[t]['max'], 2)) for t in top[:5]]} "
              f"(rel_tol {tol:.1e})")


if __name__ == "__main__":
    main()
