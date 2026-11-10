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
from functools import partial
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
LANGUAGE_COLUMNS = ("language_code", "language")
MAX_ERROR_CHARS = 200  # of each load error, in the message listing every attempt
CONTAMINATION_SOURCE = "PKU-Alignment/BeaverTails (30k_train, is_safe=False)"
CONTAMINATION_VIA = "safetune.runner.utils.data_utils.harden_contamination_pairs(n)[0]"
CONTAMINATION_GITIGNORE = "# raw harmful completions (BeaverTails): never commit or push\n*\n"


# ── loading ──────────────────────────────────────────────────────────────────


def is_hindi(row: dict) -> bool:
    """True for a row whose language_code is "hin" or whose language is "Hindi" (any case)."""
    language_code = str(row.get("language_code") or "").lower()
    language = str(row.get("language") or "").lower()
    return language_code == "hin" or language == "hindi"


def _values_are_hindi(column_names: list[str], *values) -> bool:
    """is_hindi for one row passed as column values, the way Dataset.filter passes ``input_columns``."""
    return is_hindi(dict(zip(column_names, values)))


def _hindi_rows(dataset) -> list[dict]:
    """The Hindi rows of a loaded Hub split, filtering on the language columns only (fast)."""
    columns = [column for column in LANGUAGE_COLUMNS if column in dataset.column_names]
    if not columns:
        raise KeyError(f"no language_code/language column in {dataset.column_names}")
    hindi = dataset.filter(partial(_values_are_hindi, columns), input_columns=columns)
    return [dict(row) for row in hindi]


def load_from_hub(repos: list[str]) -> tuple[list[dict], dict]:
    """The Hindi rows of the first repo whose train split loads, and a record of where they came from."""
    from datasets import load_dataset

    errors = []
    for repo in repos:
        try:
            dataset = load_dataset(repo, split="train")
            rows = _hindi_rows(dataset)
        except Exception as error:  # try the next repo
            errors.append(f"{repo}: {type(error).__name__}: {str(error)[:MAX_ERROR_CHARS]}")
            continue
        print(f"[t1] {repo}: {len(rows)} Hindi rows of {len(dataset)}")
        return rows, {
            "repo": repo,
            "split": "train",
            "n_total": len(dataset),
            "n_hindi": len(rows),
            "fingerprint": getattr(dataset, "_fingerprint", None),
        }
    raise RuntimeError("could not load the Aya Dataset:\n  " + "\n  ".join(errors))


def load_raw_rows(args, config: dict) -> tuple[list[dict], dict]:
    """(Hindi raw rows, source record) from --input-jsonl or the Hub."""
    if not args.input_jsonl:
        repos = hub_candidates(config["datasets"].get("task_t1"), *FALLBACK_REPOS)
        return load_from_hub(repos)
    raw_rows = [row for row in read_jsonl(args.input_jsonl) if is_hindi(row)]
    source = {"repo": f"local:{Path(args.input_jsonl).name}", "split": "file", "n_hindi": len(raw_rows)}
    return raw_rows, source


# ── dedup and filtering ──────────────────────────────────────────────────────


def _preprocess(raw: dict) -> dict:
    """CuratorKIT preprocessing_fn: deterministic id from the prompt; drop the annotator id."""
    row = {key: value for key, value in raw.items() if key != "user_id"}
    row["id"] = stable_id("hi", str(raw.get("inputs") or ""))
    return row


def curate(raw_path: Path, out_dir: Path) -> tuple[list[dict], dict]:
    """CuratorKIT cleaning and dedup of the Hindi rows; returns {"id", "prompt", "answer"} rows and stats."""
    from curatorkit.curator import Curator, CuratorConfig

    if out_dir.exists():
        shutil.rmtree(out_dir)
    config = CuratorConfig(
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
    result = Curator(config).run()
    rows = []
    for exported in read_jsonl(out_dir / "prompts_meta.jsonl"):
        rows.append({"id": exported["id"], "prompt": exported["prompt"], "answer": exported.get("output", "")})
    return rows, {"curatorkit_stage_counts": result.stage_counts, "n_rejected": len(result.rejected)}


def plain(raw_rows: list[dict]) -> tuple[list[dict], dict]:
    """Exact dedup on the normalised (prompt, answer) pair, in plain Python."""
    rows = []
    seen = set()
    for raw in raw_rows:
        prompt = str(raw.get("inputs") or "").strip()
        answer = str(raw.get("targets") or "").strip()
        prompt_key = normalize_for_match(prompt)
        key = (prompt_key, normalize_for_match(answer))
        if not prompt_key or key in seen:
            continue
        seen.add(key)
        rows.append({"id": stable_id("hi", prompt), "prompt": prompt, "answer": answer})
    return rows, {"n_exact_dups": len(raw_rows) - len(rows)}


def dedup(args, raw_rows: list[dict], out_dir: Path) -> tuple[list[dict], dict]:
    """Dedup by the path the flags choose. CuratorKIT reads a file, so it gets the Hindi rows as one: a
    temporary file for --input-jsonl, else the --raw-path cache."""
    if args.no_curatorkit:
        return plain(raw_rows)
    curator_dir = out_dir / "_curatorkit"
    if args.input_jsonl:
        with tempfile.TemporaryDirectory() as tmp:
            return curate(write_jsonl(Path(tmp) / "hindi.jsonl", raw_rows), curator_dir)
    return curate(write_jsonl(Path(args.raw_path), raw_rows), curator_dir)


def _has_devanagari(row: dict) -> bool:
    """True if the prompt or the answer contains Devanagari."""
    return contains_devanagari(row["prompt"]) or contains_devanagari(row["answer"])


def post_filter(rows: list[dict], require_devanagari: bool) -> tuple[list[dict], dict]:
    """Drop rows with an empty answer, (optionally) without Devanagari, or with a repeated prompt."""
    kept = []
    seen_text = set()
    seen_id = set()
    stats = {"empty_answer": 0, "no_devanagari": 0, "repeated_prompt": 0}
    for row in rows:
        key = normalize_for_match(row["prompt"])
        if not str(row["answer"]).strip():
            stats["empty_answer"] += 1
        elif require_devanagari and not _has_devanagari(row):
            stats["no_devanagari"] += 1
        elif key in seen_text or row["id"] in seen_id:
            stats["repeated_prompt"] += 1
        else:
            seen_text.add(key)
            seen_id.add(row["id"])
            kept.append({"id": row["id"], "prompt": clean_prompt(row["prompt"]), "answer": row["answer"]})
    return kept, stats


# ── heldout and train files ──────────────────────────────────────────────────


def select(rows: list[dict], sizes: list[int], heldout: int, seed: int, allow_fewer: bool) -> dict[str, list[dict]]:
    """Seeded heldout and nested train files (each smaller train file is a prefix of the largest)."""
    by_id = {row["id"]: row for row in rows}
    order = sorted(by_id)
    random.Random(seed).shuffle(order)
    needed = heldout + max(sizes)
    if len(order) < needed:
        message = f"only {len(order)} usable Hindi rows; need {needed} (heldout {heldout} + train {max(sizes)})"
        if not allow_fewer:
            raise SystemExit(message + ". Rerun with --allow-fewer to write what there is.")
        print(f"[t1] WARNING: {message}; train files will be short")
    heldout_ids = order[:heldout]
    train_pool = order[heldout:needed]
    files = {"heldout": [by_id[row_id] for row_id in heldout_ids]}
    for size in sorted(sizes, reverse=True):
        files[f"train_{size}"] = [by_id[row_id] for row_id in train_pool[:size]]
    return files


def train_sizes(args, runs: dict) -> list[int]:
    """--sizes, or else the sizes of the config's T1 runs, largest first."""
    if args.sizes:
        return args.sizes
    t1_sizes = {int(run["size"]) for run in runs.values() if run.get("task") == "T1"}
    return sorted(t1_sizes, reverse=True)


# ── contamination ────────────────────────────────────────────────────────────


def contamination_needed(runs: dict) -> int:
    """Contamination rows the config's runs need: the most any contaminated run asks for."""
    needed = 0
    for run in runs.values():
        if float(run.get("contamination", 0)) > 0:
            run_needs = contamination_count(int(run["size"]), float(run["contamination"]))
            needed = max(needed, run_needs)
    return needed


def _contamination_row(index: int, prompt: str, answer: str) -> dict:
    """One BeaverTails pair as a contamination row, with where it came from."""
    metadata = {"source": CONTAMINATION_SOURCE, "via": CONTAMINATION_VIA, "index": index, "harmful": True}
    return {"id": stable_id("en", prompt), "prompt": prompt, "answer": answer, "metadata": metadata}


def build_contamination(n: int, out_dir: Path) -> Path:
    """Write n BeaverTails harmful pairs to ``beavertails_<n>.jsonl`` in a folder git ignores entirely."""
    from safetune.runner.utils.data_utils import harden_contamination_pairs

    pairs = harden_contamination_pairs(n)[0]
    if len(pairs) < n:
        raise RuntimeError(f"harden_contamination_pairs({n}) returned only {len(pairs)} pairs")
    rows = [_contamination_row(index, prompt, answer) for index, (prompt, answer) in enumerate(pairs[:n])]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".gitignore").write_text(CONTAMINATION_GITIGNORE)
    return write_jsonl(out_dir / f"beavertails_{n}.jsonl", rows)


# ── main ─────────────────────────────────────────────────────────────────────


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-dir", default=None, help="processed data root (default: data/processed)")
    parser.add_argument("--raw-path", default=str(DEFAULT_RAW), help="where the Hindi rows are cached as JSONL")
    parser.add_argument(
        "--input-jsonl", default=None, help="local Aya Dataset rows (inputs, targets, language_code, ...)"
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=None, help="train sizes (default: T1 run sizes in the config)"
    )
    parser.add_argument("--heldout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=None, help="default: config split_seed")
    parser.add_argument("--no-curatorkit", action="store_true")
    parser.add_argument("--no-require-devanagari", action="store_true")
    parser.add_argument("--allow-fewer", action="store_true", help="write short train files instead of failing")
    parser.add_argument("--skip-contamination", action="store_true")
    parser.add_argument(
        "--contamination-n",
        type=int,
        default=None,
        help="rows for the contamination file (default: what the config's runs need)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    seed = config["split_seed"] if args.seed is None else args.seed
    runs = config.get("runs", {})
    sizes = train_sizes(args, runs)
    data_dir = resolve_data_dir(args.data_dir)
    out = data_dir / "task_t1"

    raw_rows, source = load_raw_rows(args, config)
    rows, dedup_stats = dedup(args, raw_rows, out)
    rows, filter_stats = post_filter(rows, require_devanagari=not args.no_require_devanagari)
    files = select(rows, sizes, args.heldout, seed, args.allow_fewer)
    for name, file_rows in files.items():
        write_jsonl(out / f"{name}.jsonl", file_rows)
    counts = ", ".join(f"{name}={len(file_rows)}" for name, file_rows in files.items())
    print(f"[t1] {counts} -> {out}")

    manifest = {
        "name": "task_t1",
        "built": timestamp(),
        "git": git_hash(),
        "source": source,
        "curatorkit": not args.no_curatorkit,
        **dedup_stats,
        "filters": filter_stats,
        "n_usable": len(rows),
        "seed": seed,
        "sizes": sizes,
        "heldout": args.heldout,
        "counts": {name: len(file_rows) for name, file_rows in files.items()},
    }
    if not args.skip_contamination:
        # --contamination-n 0 (like no value) means "what the config's runs need"
        n_contamination = args.contamination_n if args.contamination_n else contamination_needed(runs)
        if n_contamination > 0:
            path = build_contamination(n_contamination, data_dir / "contamination")
            manifest["contamination"] = {"n": n_contamination, "file": str(path.relative_to(data_dir))}
            print(f"[t1] contamination: {n_contamination} BeaverTails rows -> {path} (gitignored folder; never commit)")
    write_manifest(out / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
