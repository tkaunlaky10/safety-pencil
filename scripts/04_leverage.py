"""Leverage map (Prompt 4 steps 3-4): per layer and module, the largest α_i γ_j and the top-10 sum, plus
per-head lower bounds for o_proj and per-KV-group lower bounds for v_proj. Uses the same bases, row set
and thresholds as the repair.

For each language with harmful factors on disk, and for all of them pooled with equal weights, writes
outputs/results/leverage_{lang}_run{run}.json and outputs/figures/leverage_{lang}.png and
leverage_heads_{lang}.png.

    python scripts/04_leverage.py --run A
    python scripts/04_leverage.py --run A --dry-run
"""

from __future__ import annotations

import argparse

from pencil.leverage import leverage_table
from pencil.modules import LEAVES, layer_order
from pencil.plots import heatmap, log10_or_nan
from pencil.repair import ModelRepair
from pencil.results import write_result
from pencil.runtime import (
    available_langs,
    common_args,
    harm_factors,
    load_model,
    model_shape,
    prepare_factors,
    repair_kwargs,
    setup,
    task_factors,
)


def plot(table: dict, shape: dict, figures, name: str, run: str) -> None:
    modules = [[float("nan")] * len(LEAVES) for _ in range(shape["n_layers"])]
    heads = [[float("nan")] * shape["n_heads"] for _ in range(shape["n_layers"])]
    for row in table.values():
        modules[row["layer"]][LEAVES.index(row["module"])] = log10_or_nan(row["max"])
        if "per_head_lower_bound" in row:
            heads[row["layer"]] = [log10_or_nan(v) for v in row["per_head_lower_bound"]]
    heatmap(
        modules,
        [m.replace("_proj", "") for m in LEAVES],
        figures / f"leverage_{name}.png",
        title=f"leverage, harmful={name}, task=run {run}",
        colorbar="log10 max αγ",
    )
    heatmap(
        heads,
        range(shape["n_heads"]),
        figures / f"leverage_heads_{name}.png",
        xlabel="query head",
        title=f"o_proj per-head leverage, {name}\n(lower bounds on the module's top value)",
        colorbar="log10 αγ",
        width=0.3 * shape["n_heads"] + 2,
    )


def main(argv=None):
    p = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--run", default="A")
    p.add_argument("--row-set", default=None, help="default pencil.repair_row_set")
    args = p.parse_args(argv)
    ctx = setup(args)
    pc = ctx.cfg["pencil"]
    model = load_model(ctx.aligned, ctx)  # for shapes and the LayerNorm null vectors
    shape = model_shape(model)
    langs = available_langs(ctx)
    row_set = args.row_set or pc.get("repair_row_set", "all")
    for name, pool in [(lang, [lang]) for lang in langs] + ([("pooled", langs)] if len(langs) > 1 else []):
        fh, ft, rel_tol, floors = prepare_factors(model, harm_factors(ctx, pool), task_factors(ctx, args.run), pc)
        repairer = ModelRepair(fh, ft, **repair_kwargs(pc, rel_tol, row_set=row_set))
        modules = layer_order(m for m in fh.modules if m in ft.a_key)
        table = leverage_table(
            repairer, modules, n_heads=shape["n_heads"], n_kv=shape["n_kv"], head_dim=shape["head_dim"]
        )
        plot(table, shape, ctx.out("figures"), name, args.run)
        top = sorted(table, key=lambda n: table[n]["max"], reverse=True)[:10]
        write_result(
            f"leverage_{name}_run{args.run}",
            {"table": table, "top10_modules": top, "rel_tol": rel_tol, "null_floors": floors, "row_set": row_set},
            config={"pencil": pc, "langs": pool},
            checkpoint=f"run{args.run}",
            out_dir=ctx.out("results"),
        )
        print(
            f"{name}: top modules {[(t.split('layers.')[1], round(table[t]['max'], 2)) for t in top[:5]]} "
            f"(rel_tol {rel_tol:.1e})"
        )


if __name__ == "__main__":
    main()
