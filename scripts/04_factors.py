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
    task_pass_name,
    task_rows,
    theta0_answers,
    write_json,
)

# Without an explicit --layers-per-pass, a chunk's factors may use at most this share of free GPU memory.
GPU_MEMORY_SHARE = 0.6

# The flag each side cannot run without, and the error shown when it is missing.
REQUIRED_FLAG = {"harm": "lang", "task": "run", "general": "data_file"}
MISSING_FLAG_ERROR = {
    "harm": "--side harm needs --lang",
    "task": "--side task needs --run",
    "general": "--side general needs --data-file",
}


def parse_layers(spec: str, n_layers: int, selection_layers: list[int]) -> list[int]:
    """'all', 'selection' or a list like '0-5,12'."""
    if spec == "all":
        return list(range(n_layers))
    if spec == "selection":
        return [layer for layer in selection_layers if layer < n_layers]
    layers = []
    for part in spec.split(","):
        start, _, end = part.partition("-")
        if end:
            layers += list(range(int(start), int(end) + 1))
        else:
            layers.append(int(start))
    return layers


def side_batches(ctx, args, model, tokenizer, n_examples: int) -> tuple[str, str, list[dict]]:
    """(pass name, which θ, batches) for the requested side. Merges the adapter into `model` for the task side."""
    factor_cfg = ctx.cfg["factors"]
    batch_size = factor_cfg.get("batch_size", 1)
    if args.side == "harm":
        batches = theta0_answers(ctx, model, tokenizer, args.lang, "factor", n_examples, batch_size)
        return args.lang, "theta0", batches
    if args.side == "task":
        run = ctx.cfg["runs"][args.run]
        merge_deltas(model, lora_delta(adapter_dir(ctx, args.run, args.frac)))
        rows = task_rows(ctx, f"train_{run['size']}", n_examples)
        name = task_pass_name(args.run, args.frac)
        theta = f"theta1 run {args.run}"
    else:
        rows = read_jsonl(args.data_file)[:n_examples]
        name = "general"
        theta = "theta0"
    pairs = [(row["prompt"], row["answer"]) for row in rows]
    encoded = encode_pairs(tokenizer, pairs, factor_cfg["max_len"])
    return name, theta, make_batches(encoded, tokenizer.pad_token_id, batch_size)


def layers_per_pass(ctx, requested: int | None, bytes_per_layer: int) -> int:
    """The configured chunk size, reduced to what fits in 60% of free GPU memory (unless given explicitly)."""
    if requested:
        return requested
    chunk_size = ctx.cfg["factors"]["layers_per_pass"]
    if not ctx.device.startswith("cuda"):
        return chunk_size
    free_bytes = torch.cuda.mem_get_info()[0]
    fit = max(1, int(GPU_MEMORY_SHARE * free_bytes // bytes_per_layer))
    if fit < chunk_size:
        print(f"memory: {free_bytes / 1e9:.1f} GB free fits {fit} layer(s) per pass, not {chunk_size}")
        chunk_size = fit
    return chunk_size


def build_parser() -> argparse.ArgumentParser:
    """The command-line flags (see the module docstring for examples)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common_args(parser)
    parser.add_argument("--side", choices=["harm", "task", "general"], required=True)
    parser.add_argument("--lang", help="harm side: en, hi or hinglish")
    parser.add_argument("--run", help="task side: run id (A-D)")
    parser.add_argument("--frac", type=int, default=100, help="task side: the adapter at this %% of training")
    parser.add_argument("--data-file", help="general side: JSONL rows with prompt and answer")
    parser.add_argument("--layers", default="all", help="'all', 'selection' or e.g. '0-5,12'")
    parser.add_argument("--row-sets", default=None, help="comma list (default factors.row_sets)")
    parser.add_argument("--n", type=int, default=None, help="examples (default factors.n_prompts)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers-per-pass", type=int, default=None, help="override factors.layers_per_pass")
    parser.add_argument("--overwrite", action="store_true", help="replace existing factor files for this pass")
    parser.add_argument("--diag", action="store_true", help="also collect the per-sequence diagonal Fisher")
    return parser


def clear_old_factors(parser, out, overwrite: bool) -> None:
    """Delete factor files left in `out` by an earlier pass; refuse (exit with an error) unless --overwrite."""
    stale = sorted(out.glob("layers*.safetensors"))
    if stale and not overwrite:
        parser.error(f"{out} already has {len(stale)} factor files; pass --overwrite to replace them")
    for path in stale:
        path.unlink()


def collect_chunk(ctx, args, model, batches, chunk: list[int], row_sets: tuple, out, pass_info: dict):
    """One pass over `batches` with only the layers in `chunk` hooked; saves out/layers{a}-{b}.safetensors.
    Returns the finalized factors and the chunk's timing record."""
    on_cuda = ctx.device.startswith("cuda")
    t_chunk = time.time()
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    accum_dtype = getattr(torch, ctx.cfg["factors"].get("accum_dtype", "float32"))
    collector = KFACCollector(
        model,
        layers=chunk,
        row_sets=row_sets,
        seed=args.seed,
        diag=args.diag,
        diag_row_set=single_row_set(ctx.cfg["pencil"]),
        accum_dtype=accum_dtype,
    )
    with collector:
        for batch in batches:
            collector.collect(batch)
    factors = collector.finalize(meta={**pass_info, "layers": chunk, "model": ctx.aligned})
    path = out / f"layers{chunk[0]:02d}-{chunk[-1]:02d}.safetensors"
    factors.save(path)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if on_cuda else None
    seconds = time.time() - t_chunk
    message = f"  layers {chunk}: {seconds:.0f}s"
    if peak_gb:
        message += f", peak {peak_gb:.1f} GB"
    print(message)
    return factors, {"layers": chunk, "seconds": seconds, "peak_gpu_gb": peak_gb, "file": path.name}


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    side = args.side
    if not getattr(args, REQUIRED_FLAG[side]):
        parser.error(MISSING_FLAG_ERROR[side])
    ctx = setup(args)
    factor_cfg = ctx.cfg["factors"]
    row_sets = tuple(args.row_sets.split(",")) if args.row_sets else tuple(factor_cfg["row_sets"])
    tokenizer = load_tokenizer(ctx)
    model = load_model(ctx.aligned, ctx)
    shape = model_shape(model)
    t_start = time.time()
    name, theta, batches = side_batches(ctx, args, model, tokenizer, args.n or factor_cfg["n_prompts"])
    t_data = time.time() - t_start

    layers = parse_layers(args.layers, shape["n_layers"], factor_cfg["selection_layers"])
    if not layers:
        parser.error(f"--layers {args.layers!r} selects no layers (the model has {shape['n_layers']})")
    bytes_per_layer = factor_bytes(
        shape["hidden"], shape["inter"], shape["n_heads"], shape["n_kv"], shape["head_dim"], n_row_sets=len(row_sets)
    )
    chunk_size = layers_per_pass(ctx, args.layers_per_pass, bytes_per_layer)
    n_seqs = sum(batch["input_ids"].shape[0] for batch in batches)
    print(
        f"{side}/{name}: {n_seqs} sequences, layers {layers}, row sets {row_sets}; "
        f"{bytes_per_layer / 1e9:.2f} GB per layer, {chunk_size} layer(s) per pass"
    )

    out = factor_dir(ctx, side, name, selection=args.layers == "selection")
    out.mkdir(parents=True, exist_ok=True)
    clear_old_factors(parser, out, args.overwrite)

    pass_info = {"side": side, "name": name, "theta": theta}
    timing = []
    for start in range(0, len(layers), chunk_size):
        chunk = layers[start : start + chunk_size]
        factors, chunk_timing = collect_chunk(ctx, args, model, batches, chunk, row_sets, out, pass_info)
        timing.append(chunk_timing)

    # n_loss and n_rows are the same for every chunk (same batches, same labels); take the last chunk's.
    meta = {
        "side": side,
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
