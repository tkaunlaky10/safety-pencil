"""K-FAC factor passes on Tiny Aya (Prompt 4 step 2).

Sides:
  harm     at θ0, per language: factor-split harmful prompts with θ0's own greedy answers teacher-forced
           (θ0 is the aligned model; we never generate new harmful completions)
  task     at θ1 = θ0 + ΔW of a run's adapter (merged in memory): training rows with their answers
  general  at θ0 on general instruction rows (--data-file), for the standard Fisher merge control

Layers are hooked a chunk at a time (one pass over the data per chunk, same sampling seed, so every chunk
sees the same labels). Output: outputs/factors/{side}/{name}/layers{a}-{b}.safetensors and meta.json;
--layers selection writes to outputs/factors_selection/ so the two passes never mix.

Factors are fp32 on disk: about 36 × (3·d_mlp² + 5·2048² + 2·512²) × 4 bytes per side and row set, ~52 GB at
d_mlp = 11008. Run the full pass with the one or two row sets the selection pass chose.

    python scripts/04_factors.py --side harm --lang en --layers selection      # row-set selection pass
    python scripts/04_factors.py --side task --run A --layers selection
    python scripts/04_factors.py --side harm --lang hi --row-sets full,full_reduce
    python scripts/04_factors.py --side harm --lang en --dry-run
"""

from __future__ import annotations

import argparse
import time

import torch

from pencil.datasets import read_jsonl
from pencil.delta import lora_delta, merge_deltas
from pencil.factors import KFACCollector, factor_bytes
from pencil.runtime import (
    adapter_dir,
    common_args,
    encode_pairs,
    factor_dir,
    load_model,
    load_tokenizer,
    make_batches,
    model_shape,
    setup,
    single_row_set,
    task_rows,
    theta0_answers,
    write_json,
)


def parse_layers(spec: str, n_layers: int, selection_layers: list[int]) -> list[int]:
    """'all', 'selection' or a list like '0-5,12'."""
    if spec == "all":
        return list(range(n_layers))
    if spec == "selection":
        return [layer for layer in selection_layers if layer < n_layers]
    layers = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        layers += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return layers


def side_batches(ctx, args, model, tok, n: int) -> tuple[str, str, list[dict]]:
    """(pass name, which θ, batches) for the requested side. Merges the adapter into `model` for the task side."""
    fc = ctx.cfg["factors"]
    size = fc.get("batch_size", 1)
    if args.side == "harm":
        return args.lang, "theta0", theta0_answers(ctx, model, tok, args.lang, "factor", n, size)
    if args.side == "task":
        run = ctx.cfg["runs"][args.run]
        merge_deltas(model, lora_delta(adapter_dir(ctx, args.run, args.frac)))
        rows = task_rows(ctx, f"train_{run['size']}", n)
        name = f"run{args.run}" + ("" if args.frac == 100 else f"_frac{args.frac}")
        theta = f"theta1 run {args.run}"
    else:
        rows, name, theta = read_jsonl(args.data_file)[:n], "general", "theta0"
    encoded = encode_pairs(tok, [(r["prompt"], r["answer"]) for r in rows], fc["max_len"])
    return name, theta, make_batches(encoded, tok.pad_token_id, size)


def layers_per_pass(ctx, requested: int | None, bytes_per_layer: int) -> int:
    """The configured chunk size, reduced to what fits in 60% of free GPU memory (unless given explicitly)."""
    k = requested or ctx.cfg["factors"]["layers_per_pass"]
    if ctx.device.startswith("cuda") and not requested:
        free = torch.cuda.mem_get_info()[0]
        fit = max(1, int(0.6 * free // bytes_per_layer))
        if fit < k:
            print(f"memory: {free / 1e9:.1f} GB free fits {fit} layer(s) per pass, not {k}")
            k = fit
    return k


def main(argv=None):
    p = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--side", choices=["harm", "task", "general"], required=True)
    p.add_argument("--lang", help="harm side: en, hi or hinglish")
    p.add_argument("--run", help="task side: run id (A-D)")
    p.add_argument("--frac", type=int, default=100, help="task side: the adapter at this %% of training")
    p.add_argument("--data-file", help="general side: JSONL rows with prompt and answer")
    p.add_argument("--layers", default="all", help="'all', 'selection' or e.g. '0-5,12'")
    p.add_argument("--row-sets", default=None, help="comma list (default factors.row_sets)")
    p.add_argument("--n", type=int, default=None, help="examples (default factors.n_prompts)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layers-per-pass", type=int, default=None, help="override factors.layers_per_pass")
    p.add_argument("--overwrite", action="store_true", help="replace existing factor files for this pass")
    p.add_argument("--diag", action="store_true", help="also collect the per-sequence diagonal Fisher")
    args = p.parse_args(argv)
    needs = {"harm": args.lang, "task": args.run, "general": args.data_file}[args.side]
    if not needs:
        p.error(
            {
                "harm": "--side harm needs --lang",
                "task": "--side task needs --run",
                "general": "--side general needs --data-file",
            }[args.side]
        )
    ctx = setup(args)
    fc = ctx.cfg["factors"]
    row_sets = tuple(args.row_sets.split(",")) if args.row_sets else tuple(fc["row_sets"])
    tok = load_tokenizer(ctx)
    model = load_model(ctx.aligned, ctx)
    shape = model_shape(model)
    t_start = time.time()
    name, theta, batches = side_batches(ctx, args, model, tok, args.n or fc["n_prompts"])
    t_data = time.time() - t_start

    layers = parse_layers(args.layers, shape["n_layers"], fc["selection_layers"])
    bytes_per_layer = factor_bytes(
        shape["hidden"], shape["inter"], shape["n_heads"], shape["n_kv"], shape["head_dim"], n_row_sets=len(row_sets)
    )
    k = layers_per_pass(ctx, args.layers_per_pass, bytes_per_layer)
    n_seqs = sum(b["input_ids"].shape[0] for b in batches)
    print(
        f"{args.side}/{name}: {n_seqs} sequences, layers {layers}, row sets {row_sets}; "
        f"{bytes_per_layer / 1e9:.2f} GB per layer, {k} layer(s) per pass"
    )

    out = factor_dir(ctx, args.side, name, selection=args.layers == "selection")
    out.mkdir(parents=True, exist_ok=True)
    stale = sorted(out.glob("layers*.safetensors"))
    if stale and not args.overwrite:
        p.error(f"{out} already has {len(stale)} factor files; pass --overwrite to replace them")
    for f in stale:
        f.unlink()

    diag_rows = single_row_set(ctx.cfg["pencil"])
    timing = []
    for chunk in [layers[i : i + k] for i in range(0, len(layers), k)]:
        t0 = time.time()
        if ctx.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        with KFACCollector(
            model,
            layers=chunk,
            row_sets=row_sets,
            seed=args.seed,
            diag=args.diag,
            diag_row_set=diag_rows,
            accum_dtype=getattr(torch, fc.get("accum_dtype", "float32")),
        ) as kc:
            for b in batches:
                kc.collect(b)
        factors = kc.finalize(
            meta={"side": args.side, "name": name, "theta": theta, "layers": chunk, "model": ctx.aligned}
        )
        path = out / f"layers{chunk[0]:02d}-{chunk[-1]:02d}.safetensors"
        factors.save(path)
        peak = torch.cuda.max_memory_allocated() / 1e9 if ctx.device.startswith("cuda") else None
        timing.append({"layers": chunk, "seconds": time.time() - t0, "peak_gpu_gb": peak, "file": path.name})
        print(f"  layers {chunk}: {timing[-1]['seconds']:.0f}s" + (f", peak {peak:.1f} GB" if peak else ""))

    meta = {
        "side": args.side,
        "name": name,
        "theta": theta,
        "n_sequences": n_seqs,
        "n_loss": factors.n_loss,
        "n_rows": factors.n_rows,
        "row_sets": row_sets,
        "seed": args.seed,
        "data_s": t_data,
        "chunks": timing,
        "total_s": time.time() - t_start,
        "bytes_per_layer": bytes_per_layer,
        "shape": shape,
    }
    write_json(out / "meta.json", meta)
    print(f"done in {meta['total_s']:.0f}s -> {out}")


if __name__ == "__main__":
    main()
