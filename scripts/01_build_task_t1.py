#!/usr/bin/env python
"""Build task T1 (Hindi rows of the Aya Dataset) and the run-D contamination file.

1. Load the Aya Dataset train split (CohereLabs/aya_dataset, then CohereForAI/aya_dataset), or a local
   JSONL with --input-jsonl, and keep the Hindi rows (language_code "hin" or language "Hindi").
2. CuratorKIT: ``field_mapping={"inputs": "instruction", "targets": "output"}`` (without it every row is
   rejected), deterministic ids via ``preprocessing_fn``, min_tokens=1, exact + MinHash dedup,
   ``export_formats=["prompts_meta"]``. ``--no-curatorkit`` does exact dedup in Python instead. Then drop
   rows with an empty answer, with no Devanagari in prompt or answer (``--no-require-devanagari`` keeps
   them), and repeated prompts.
3. Shuffle the ids with the seed: the first ``--heldout`` (300) rows are ``heldout.jsonl``; the next
   max(--sizes) rows are ``train_<max>.jsonl``, and each smaller ``train_<n>.jsonl`` is its first n rows
   (so train_1000 is a subset of train_4000). Rows: ``{"id", "prompt", "answer"}``.
4. Contamination for run D: ``n = round(frac * size / (1 - frac))`` rows (20 for 1000 rows at 2%) from
   SafeTune's ``harden_contamination_pairs(n)[0]`` (BeaverTails 30k_train) ->
   ``contamination/beavertails_<n>.jsonl``. These are existing BeaverTails completions; nothing is
   generated. The folder gets a ``.gitignore`` so they are never committed.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import tempfile
from pathlib import Path

from pencil.config import load_config
from pencil.datasets import (
    REPO_ROOT,
    clean_prompt,
    contains_devanagari,
    contamination_count,
    hub_candidates,
    normalize_for_match,
    read_jsonl,
    resolve_data_dir,
    stable_id,
    timestamp,
    write_jsonl,
    write_manifest,
)
from pencil.results import git_hash

FALLBACK_REPOS = ("CohereLabs/aya_dataset", "CohereForAI/aya_dataset")
DEFAULT_RAW = REPO_ROOT / "data" / "raw" / "aya_dataset_hindi.jsonl"  # gitignored


def is_hindi(row: dict) -> bool:
    return str(row.get("language_code") or "").lower() == "hin" or str(row.get("language") or "").lower() == "hindi"


def _hindi_rows(ds) -> list[dict]:
    """The Hindi rows of a loaded Hub split, filtering on the language columns only (fast)."""
    cols = [c for c in ("language_code", "language") if c in ds.column_names]
    if not cols:
        raise KeyError(f"no language_code/language column in {ds.column_names}")
    return [dict(r) for r in ds.filter(lambda *vals: is_hindi(dict(zip(cols, vals))), input_columns=cols)]


def load_from_hub(repos: list[str]) -> tuple[list[dict], dict]:
    from datasets import load_dataset

    errors = []
    for repo in repos:
        try:
            ds = load_dataset(repo, split="train")
            rows = _hindi_rows(ds)
        except Exception as e:  # try the next repo
            errors.append(f"{repo}: {type(e).__name__}: {str(e)[:200]}")
            continue
        print(f"[t1] {repo}: {len(rows)} Hindi rows of {len(ds)}")
        return rows, {
            "repo": repo,
            "split": "train",
            "n_total": len(ds),
            "n_hindi": len(rows),
            "fingerprint": getattr(ds, "_fingerprint", None),
        }
    raise RuntimeError("could not load the Aya Dataset:\n  " + "\n  ".join(errors))


def _preprocess(raw: dict) -> dict:
    """CuratorKIT preprocessing_fn: deterministic id from the prompt; drop the annotator id."""
    row = {k: v for k, v in raw.items() if k != "user_id"}
    row["id"] = stable_id("hi", str(raw.get("inputs") or ""))
    return row


def curate(raw_path: Path, out_dir: Path) -> tuple[list[dict], dict]:
    from curatorkit.curator import Curator, CuratorConfig

    if out_dir.exists():
        shutil.rmtree(out_dir)
    cfg = CuratorConfig(
        dataset=str(raw_path),
        field_mapping={"inputs": "instruction", "targets": "output"},
        preprocessing_fn=_preprocess,
        min_tokens=1,
        dedup="minhash",
        toxicity_gate=False,
        export_formats=["prompts_meta"],
        output_dir=str(out_dir),
        name="aya_dataset_hindi_t1",
        write_hf_readme=False,
    )
    result = Curator(cfg).run()
    rows = [
        {"id": r["id"], "prompt": r["prompt"], "answer": r.get("output", "")}
        for r in read_jsonl(out_dir / "prompts_meta.jsonl")
    ]
    return rows, {"curatorkit_stage_counts": result.stage_counts, "n_rejected": len(result.rejected)}


def plain(raw_rows: list[dict]) -> tuple[list[dict], dict]:
    """Exact dedup on the normalised (prompt, answer) pair, in plain Python."""
    rows, seen = [], set()
    for r in raw_rows:
        prompt, answer = str(r.get("inputs") or "").strip(), str(r.get("targets") or "").strip()
        key = (normalize_for_match(prompt), normalize_for_match(answer))
        if not key[0] or key in seen:
            continue
        seen.add(key)
        rows.append({"id": stable_id("hi", prompt), "prompt": prompt, "answer": answer})
    return rows, {"n_exact_dups": len(raw_rows) - len(rows)}


def post_filter(rows: list[dict], require_devanagari: bool) -> tuple[list[dict], dict]:
    """Drop rows with an empty answer, (optionally) without Devanagari, or with a repeated prompt."""
    out, seen_text, seen_id = [], set(), set()
    stats = {"empty_answer": 0, "no_devanagari": 0, "repeated_prompt": 0}
    for r in rows:
        key = normalize_for_match(r["prompt"])
        if not str(r["answer"]).strip():
            stats["empty_answer"] += 1
        elif require_devanagari and not (contains_devanagari(r["prompt"]) or contains_devanagari(r["answer"])):
            stats["no_devanagari"] += 1
        elif key in seen_text or r["id"] in seen_id:
            stats["repeated_prompt"] += 1
        else:
            seen_text.add(key)
            seen_id.add(r["id"])
            out.append({"id": r["id"], "prompt": clean_prompt(r["prompt"]), "answer": r["answer"]})
    return out, stats


def select(rows: list[dict], sizes: list[int], heldout: int, seed: int, allow_fewer: bool) -> dict[str, list[dict]]:
    """Seeded heldout and nested train files (each smaller train file is a prefix of the largest)."""
    by_id = {r["id"]: r for r in rows}
    order = sorted(by_id)
    random.Random(seed).shuffle(order)
    need = heldout + max(sizes)
    if len(order) < need:
        msg = f"only {len(order)} usable Hindi rows; need {need} (heldout {heldout} + train {max(sizes)})"
        if not allow_fewer:
            raise SystemExit(msg + ". Rerun with --allow-fewer to write what there is.")
        print(f"[t1] WARNING: {msg}; train files will be short")
    held, pool = order[:heldout], order[heldout:need]
    out = {"heldout": [by_id[i] for i in held]}
    for s in sorted(sizes, reverse=True):
        out[f"train_{s}"] = [by_id[i] for i in pool[:s]]
    return out


def contamination_needed(runs: dict) -> int:
    """Contamination rows the config's runs need: the most any contaminated run asks for."""
    return max(
        (
            contamination_count(int(r["size"]), float(r["contamination"]))
            for r in runs.values()
            if float(r.get("contamination", 0)) > 0
        ),
        default=0,
    )


def build_contamination(n: int, out_dir: Path) -> Path:
    from safetune.runner.utils.data_utils import harden_contamination_pairs

    pairs = harden_contamination_pairs(n)[0]
    if len(pairs) < n:
        raise RuntimeError(f"harden_contamination_pairs({n}) returned only {len(pairs)} pairs")
    rows = [
        {
            "id": stable_id("en", p),
            "prompt": p,
            "answer": a,
            "metadata": {
                "source": "PKU-Alignment/BeaverTails (30k_train, is_safe=False)",
                "via": "safetune.runner.utils.data_utils.harden_contamination_pairs(n)[0]",
                "index": k,
                "harmful": True,
            },
        }
        for k, (p, a) in enumerate(pairs[:n])
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".gitignore").write_text("# raw harmful completions (BeaverTails): never commit or push\n*\n")
    return write_jsonl(out_dir / f"beavertails_{n}.jsonl", rows)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__
    )
    p.add_argument("--config", default=None)
    p.add_argument("--data-dir", default=None, help="processed data root (default: data/processed)")
    p.add_argument("--raw-path", default=str(DEFAULT_RAW), help="where the Hindi rows are cached as JSONL")
    p.add_argument("--input-jsonl", default=None, help="local Aya Dataset rows (inputs, targets, language_code, ...)")
    p.add_argument(
        "--sizes", type=int, nargs="+", default=None, help="train sizes (default: T1 run sizes in the config)"
    )
    p.add_argument("--heldout", type=int, default=300)
    p.add_argument("--seed", type=int, default=None, help="default: config split_seed")
    p.add_argument("--no-curatorkit", action="store_true")
    p.add_argument("--no-require-devanagari", action="store_true")
    p.add_argument("--allow-fewer", action="store_true", help="write short train files instead of failing")
    p.add_argument("--skip-contamination", action="store_true")
    p.add_argument(
        "--contamination-n",
        type=int,
        default=None,
        help="rows for the contamination file (default: what the config's runs need)",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    seed = cfg["split_seed"] if args.seed is None else args.seed
    runs = cfg.get("runs", {})
    sizes = args.sizes or sorted({int(r["size"]) for r in runs.values() if r.get("task") == "T1"}, reverse=True)
    data_dir = resolve_data_dir(args.data_dir)
    out = data_dir / "task_t1"

    if args.input_jsonl:
        raw_rows = [r for r in read_jsonl(args.input_jsonl) if is_hindi(r)]
        source = {"repo": f"local:{Path(args.input_jsonl).name}", "split": "file", "n_hindi": len(raw_rows)}
    else:
        raw_rows, source = load_from_hub(hub_candidates(cfg["datasets"].get("task_t1"), *FALLBACK_REPOS))
    if args.no_curatorkit:
        rows, stats = plain(raw_rows)
    elif args.input_jsonl:  # CuratorKIT reads a file: hand it the Hindi-filtered rows
        with tempfile.TemporaryDirectory() as tmp:
            rows, stats = curate(write_jsonl(Path(tmp) / "hindi.jsonl", raw_rows), out / "_curatorkit")
    else:
        rows, stats = curate(write_jsonl(Path(args.raw_path), raw_rows), out / "_curatorkit")
    rows, filt = post_filter(rows, require_devanagari=not args.no_require_devanagari)
    parts = select(rows, sizes, args.heldout, seed, args.allow_fewer)
    for name, part in parts.items():
        write_jsonl(out / f"{name}.jsonl", part)
    print("[t1] " + ", ".join(f"{k}={len(v)}" for k, v in parts.items()) + f" -> {out}")

    manifest = {
        "name": "task_t1",
        "built": timestamp(),
        "git": git_hash(),
        "source": source,
        "curatorkit": not args.no_curatorkit,
        **stats,
        "filters": filt,
        "n_usable": len(rows),
        "seed": seed,
        "sizes": sizes,
        "heldout": args.heldout,
        "counts": {k: len(v) for k, v in parts.items()},
    }
    if not args.skip_contamination:
        n = args.contamination_n or contamination_needed(runs)
        if n > 0:
            path = build_contamination(n, data_dir / "contamination")
            manifest["contamination"] = {"n": n, "file": str(path.relative_to(data_dir))}
            print(f"[t1] contamination: {n} BeaverTails rows -> {path} (gitignored folder; never commit)")
    write_manifest(out / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
