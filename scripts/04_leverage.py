"""Leverage map (Prompt 4 steps 3-4): per layer and module, the largest α_i γ_j and the top-10 sum, plus
per-head lower bounds for o_proj and per-KV-group lower bounds for v_proj. Uses the same bases, row set
and thresholds as the repair.

For each language with harmful factors on disk, and for all of them pooled with equal weights, writes
outputs/results/leverage_{lang}_run{run}.json and the figures outputs/figures/leverage_{lang}_run{run}.png
and leverage_heads_{lang}_run{run}.png.

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

N_TOP_MODULES = 10  # modules listed in the result JSON
N_PRINTED_MODULES = 5  # of those, modules printed to the console


def nan_grid(n_rows: int, n_columns: int) -> list[list[float]]:
    """An n_rows × n_columns grid of NaN (blank heatmap cells)."""
    return [[float("nan")] * n_columns for _ in range(n_rows)]


def plot(table: dict, shape: dict, figures, name: str, run: str) -> None:
    """Two heatmaps of log10 leverage by layer: the module maximum, and o_proj's per-head lower bounds."""
    module_grid = nan_grid(shape["n_layers"], len(LEAVES))
    head_grid = nan_grid(shape["n_layers"], shape["n_heads"])
    for row in table.values():
        module_grid[row["layer"]][LEAVES.index(row["module"])] = log10_or_nan(row["max"])
        if "per_head_lower_bound" in row:
            head_grid[row["layer"]] = [log10_or_nan(value) for value in row["per_head_lower_bound"]]

    module_labels = [module.replace("_proj", "") for module in LEAVES]
    heatmap(
        module_grid,
        module_labels,
        figures / f"leverage_{name}_run{run}.png",
        title=f"leverage, harmful={name}, task=run {run}",
        colorbar="log10 max αγ",
    )
    heatmap(
        head_grid,
        range(shape["n_heads"]),
        figures / f"leverage_heads_{name}_run{run}.png",
        xlabel="query head",
        title=f"o_proj per-head leverage, {name}\n(lower bounds on the module's top value)",
        colorbar="log10 αγ",
        width=0.3 * shape["n_heads"] + 2,
    )


def factor_pools(langs: list[str]) -> list[tuple[str, list[str]]]:
    """(name, languages) for each language on its own, plus "pooled" over all of them when there are several."""
    pools = [(lang, [lang]) for lang in langs]
    if len(langs) > 1:
        pools.append(("pooled", langs))
    return pools


def top_modules(table: dict, count: int) -> list[str]:
    """The `count` modules with the largest leverage (max α_i γ_j), largest first."""
    return sorted(table, key=lambda module: table[module]["max"], reverse=True)[:count]


def short_summary(table: dict, modules: list[str]) -> list[tuple[str, float]]:
    """(module name after 'layers.', leverage rounded to 2 decimals) for printing."""
    summary = []
    for module in modules:
        short_name = module.split("layers.")[1]
        summary.append((short_name, round(table[module]["max"], 2)))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common_args(parser)
    parser.add_argument("--run", default="A")
    parser.add_argument("--row-set", default=None, help="default pencil.repair_row_set")
    args = parser.parse_args(argv)
    ctx = setup(args)
    pencil_cfg = ctx.cfg["pencil"]
    model = load_model(ctx.aligned, ctx)  # for shapes and the LayerNorm null vectors
    shape = model_shape(model)
    langs = available_langs(ctx)
    row_set = args.row_set or pencil_cfg.get("repair_row_set", "all")

    for name, pool in factor_pools(langs):
        pooled_harm = harm_factors(ctx, pool)
        run_task = task_factors(ctx, args.run)
        harm, task, rel_tol, floors = prepare_factors(model, pooled_harm, run_task, pencil_cfg)
        repairer = ModelRepair(harm, task, **repair_kwargs(pencil_cfg, rel_tol, row_set=row_set))
        modules = layer_order(module for module in harm.modules if module in task.a_key)
        table = leverage_table(
            repairer, modules, n_heads=shape["n_heads"], n_kv=shape["n_kv"], head_dim=shape["head_dim"]
        )
        plot(table, shape, ctx.out("figures"), name, args.run)
        top = top_modules(table, N_TOP_MODULES)
        write_result(
            f"leverage_{name}_run{args.run}",
            {"table": table, "top10_modules": top, "rel_tol": rel_tol, "null_floors": floors, "row_set": row_set},
            config={"pencil": pencil_cfg, "langs": pool},
            checkpoint=f"run{args.run}",
            out_dir=ctx.out("results"),
        )
        print(f"{name}: top modules {short_summary(table, top[:N_PRINTED_MODULES])} (rel_tol {rel_tol:.1e})")


if __name__ == "__main__":
    main()
