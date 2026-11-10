"""K-FAC factor passes on Tiny Aya (Prompt 4 step 2).

Sides:
  harm     at θ0, per language: factor-split harmful prompts with θ0's own greedy response teacher-forced
           (taken from the generation cache, generated once if missing). Never new harmful completions:
           θ0 is the aligned model, and its responses are what we want to preserve.
  task     at θ1 = θ0 + ΔW of a run's adapter (merged in memory): training rows with their answers.
  general  at θ0 on general instruction rows (--data-file), for the standard Fisher merge control.

Layers are hooked in chunks (factors.layers_per_pass), one forward/backward pass over the data per chunk,
with the same sampling seed so every chunk sees the same sampled labels. Output:
  outputs/factors/{side}/{name}/layers{a}-{b}.safetensors  +  meta.json (counts, timing, memory plan)
  (--layers selection writes to outputs/factors_selection/ instead, so the two passes never mix)

Factors are saved in fp32. Disk per side and row set is about 36 × (3·d_mlp² + 5·2048² + 2·512²) × 4 bytes,
about 52 GB at d_mlp = 11008: run the full pass with one or two row sets, chosen on the selection pass.

    python scripts/04_factors.py --side harm --lang hi
    python scripts/04_factors.py --side harm --lang en --layers selection      # row-set selection pass
    python scripts/04_factors.py --side task --run A --layers selection
    python scripts/04_factors.py --side harm --lang en --dry-run
"""
from __future__ import annotations

import argparse
import time

import torch

from pencil.delta import lora_delta, merge_deltas
from pencil.factors import KFACCollector, factor_bytes, layer_chunks
from pencil.generate import generate_cached
from pencil.datasets import read_jsonl
from pencil.runtime import (common_args, encode_pairs, factor_dir, harmful_rows, load_model, load_tokenizer,
                            make_batches, model_shape, setup, task_rows, write_json)


def parse_layers(spec: str, n_layers: int, cfg: dict) -> list[int]:
    if spec == "all":
        return list(range(n_layers))
    if spec == "selection":
        return [l for l in cfg["factors"]["selection_layers"] if l < n_layers]
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def adapter_dir(ctx, run: str, frac: int):
    return ctx.out_root / "ckpt" / run / f"frac{frac}"


def main(argv=None):
    p = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--side", choices=["harm", "task", "general"], required=True)
    p.add_argument("--lang", help="harm side: en, hi or hinglish")
    p.add_argument("--run", help="task side: run id (A-D)")
    p.add_argument("--frac", type=int, default=100, help="task side: adapter at this % of training")
    p.add_argument("--data-file", help="general side: JSONL rows with prompt and answer")
    p.add_argument("--layers", default="all", help="'all', 'selection' or e.g. '0-5,12'")
    p.add_argument("--row-sets", default=None, help="comma list (default factors.row_sets)")
    p.add_argument("--n", type=int, default=None, help="examples (default factors.n_prompts)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layers-per-pass", type=int, default=None, help="override factors.layers_per_pass")
    p.add_argument("--overwrite", action="store_true", help="replace existing factor files for this pass")
    p.add_argument("--diag", action="store_true", help="also accumulate the per-sequence diagonal Fisher (diag_fisher baseline)")
    args = p.parse_args(argv)
    ctx = setup(args)
    fc = ctx.cfg["factors"]
    n = args.n or fc["n_prompts"]
    row_sets = tuple(args.row_sets.split(",")) if args.row_sets else tuple(fc["row_sets"])
    tok = load_tokenizer(ctx)
    model = load_model(ctx.aligned, ctx)
    shape = model_shape(model)
    t_start = time.time()

    if args.side == "harm":
        if not args.lang:
            p.error("--side harm needs --lang")
        rows = harmful_rows(ctx, args.lang, "factor", n)
        gen = ctx.cfg["generation"]
        resp = generate_cached(model, tok, {f"factor_harmful_{args.lang}": rows}, checkpoint=ctx.aligned,
                               max_new_tokens=gen["max_new_tokens"], batch_size=gen["batch_size"],
                               root=ctx.out("generations"))[f"factor_harmful_{args.lang}"]
        pairs = [(r["prompt"], resp[r["id"]]) for r in rows]
        name, theta = args.lang, "theta0"
    elif args.side == "task":
        if not args.run:
            p.error("--side task needs --run")
        run = ctx.cfg["runs"][args.run]
        rows = task_rows(ctx, f"train_{run['size']}", n)
        pairs = [(r["prompt"], r["answer"]) for r in rows]
        merge_deltas(model, lora_delta(adapter_dir(ctx, args.run, args.frac)))
        name, theta = f"run{args.run}" + ("" if args.frac == 100 else f"_frac{args.frac}"), f"theta1 run {args.run}"
    else:
        if not args.data_file:
            p.error("--side general needs --data-file")
        rows = read_jsonl(args.data_file)[:n]
        pairs = [(r["prompt"], r["answer"]) for r in rows]
        name, theta = "general", "theta0"
    t_gen = time.time() - t_start

    enc = encode_pairs(tok, pairs, fc["max_len"])
    batches = make_batches(enc, tok.pad_token_id, fc.get("batch_size", 1))
    layers = parse_layers(args.layers, shape["n_layers"], ctx.cfg)
    per_layer = factor_bytes(shape["hidden"], shape["inter"], shape["n_heads"], shape["n_kv"], shape["head_dim"],
                             n_row_sets=len(row_sets))
    k = args.layers_per_pass or fc["layers_per_pass"]
    if ctx.device.startswith("cuda") and not args.layers_per_pass:
        free = torch.cuda.mem_get_info()[0]
        fit = max(1, int(0.6 * free // per_layer))           # leave room for activations and gradients
        if fit < k:
            print(f"memory: {free / 1e9:.1f} GB free fits {fit} layer(s) per pass, not {k}")
            k = fit
    chunks = [layers[i:i + k] for i in range(0, len(layers), k)]
    print(f"{args.side}/{name}: {len(enc)} sequences ({len(pairs) - len(enc)} skipped), layers {layers}, "
          f"row sets {row_sets}; factors {per_layer / 1e9:.2f} GB per layer, "
          f"{per_layer * k / 1e9:.2f} GB per pass ({k} layers)")
    accum = getattr(torch, fc.get("accum_dtype", "float32"))
    out = factor_dir(ctx, args.side, name, selection=args.layers == "selection")
    out.mkdir(parents=True, exist_ok=True)
    stale = sorted(out.glob("layers*.safetensors"))
    if stale and not args.overwrite:
        p.error(f"{out} already has factor files ({len(stale)}); pass --overwrite to replace them")
    for f in stale:
        f.unlink()
    timing = []
    for chunk in chunks:
        t0 = time.time()
        if ctx.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        with KFACCollector(model, layers=chunk, row_sets=row_sets, seed=args.seed, accum_dtype=accum,
                           diag=args.diag, diag_row_set=ctx.cfg["pencil"].get("repair_row_set", "all")
                           if isinstance(ctx.cfg["pencil"].get("repair_row_set"), str) else "all") as kc:
            for b in batches:
                kc.collect(b)
        f = kc.finalize(meta={"side": args.side, "name": name, "theta": theta, "layers": chunk,
                              "model": ctx.aligned})
        path = out / f"layers{chunk[0]:02d}-{chunk[-1]:02d}.safetensors"
        f.save(path)
        peak = torch.cuda.max_memory_allocated() / 1e9 if ctx.device.startswith("cuda") else None
        timing.append({"layers": chunk, "seconds": time.time() - t0, "peak_gpu_gb": peak, "file": path.name})
        print(f"  layers {chunk}: {timing[-1]['seconds']:.0f}s" + (f", peak {peak:.1f} GB" if peak else ""))
        n_loss, n_rows = f.n_loss, f.n_rows
        del kc, f
    meta = {"side": args.side, "name": name, "theta": theta, "n_sequences": len(enc), "n_loss": n_loss,
            "n_rows": n_rows,
            "row_sets": row_sets, "seed": args.seed, "generation_s": t_gen, "chunks": timing,
            "total_s": time.time() - t_start, "bytes_per_layer": per_layer, "shape": shape}
    write_json(out / "meta.json", meta)
    print(f"done in {meta['total_s']:.0f}s -> {out}")


if __name__ == "__main__":
    main()
