#!/usr/bin/env python
"""Build the over-refusal set: XSTest's safe prompts in en, plus hi and hinglish via an LLM.

1. Load XSTest from the first source that works (``--source``, config datasets.xstest, walledai/XSTest,
   natolambert/xstest-v2-copy, Paul/XSTest) and keep the safe prompts: label "safe", or, without a label
   column, a type not starting with "contrast". The source used is printed and recorded. Official XSTest
   v2 has 250 safe prompts (25 for each of 10 types); any other count is printed as a warning.
2. Split dev/test 20/80 with ``split_ids`` (seeded, stratified by XSTest type) into ``xstest/en/``.
3. Unless ``--skip-llm``: translate the English rows to Hindi (``HindiTranslationTask``) and convert them to
   Hinglish (``HinglishTask``) through the same CuratorKIT path as 01_make_hinglish.py, giving
   ``xstest/{hi,hinglish}/{split}.jsonl`` (ids ``<en_id>-hi`` / ``<en_id>-hinglish``) and review samples in
   outputs/review/{hi,hinglish}_check_xstest.csv.

``--dry-run`` uses the fake LLM and writes under outputs/dry_run/processed unless --data-dir is set.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from collections import Counter
from pathlib import Path

from pencil.config import load_config
from pencil.datasets import (
    hub_candidates,
    normalize_for_match,
    normalize_text,
    read_jsonl,
    resolve_data_dir,
    split_ids,
    stable_id,
    timestamp,
    write_jsonl,
    write_manifest,
)
from pencil.results import git_hash

CANDIDATES = ("walledai/XSTest", "natolambert/xstest-v2-copy", "Paul/XSTest")
SPLIT_PREFERENCE = ("test", "train", "prompts")
SPLIT_RANK = {split: rank for rank, split in enumerate(SPLIT_PREFERENCE)}
COMPLETION_COLUMNS = ("completion", "response")  # splits with model outputs, not just the prompts
EXPECTED_SAFE = 250
MAX_ERROR_CHARS = 200  # of each load error, in the message listing every attempt


def _load_hinglish_module():
    """scripts/01_make_hinglish.py, whose file name cannot be imported normally."""
    path = Path(__file__).resolve().with_name("01_make_hinglish.py")
    spec = importlib.util.spec_from_file_location("make_hinglish", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_hinglish = _load_hinglish_module()


# ── loading the safe prompts ─────────────────────────────────────────────────


def is_safe(row: dict) -> bool:
    """XSTest's label when it is "safe" or "unsafe"; otherwise a type that is set and not a contrast type."""
    label = str(row.get("label") or "").strip().lower()
    if label in ("safe", "unsafe"):
        return label == "safe"
    xstest_type = str(row.get("type") or "").strip().lower()
    return bool(xstest_type) and not xstest_type.startswith("contrast")


def pick_safe(rows: list[dict]) -> list[dict]:
    """Safe rows, deduplicated by normalised prompt (keeps the first)."""
    safe = []
    seen = set()
    for row in rows:
        key = normalize_for_match(str(row.get("prompt") or ""))
        if is_safe(row) and key and key not in seen:
            seen.add(key)
            safe.append(row)
    return safe


def _splits_in_preference_order(dataset_dict) -> list[str]:
    """Split names of a DatasetDict: splits without model completions first, then test, train, prompts,
    then the rest by name."""
    ranked = []
    for split in dataset_dict:
        columns = dataset_dict[split].column_names
        has_completions = any(column in columns for column in COMPLETION_COLUMNS)
        preference = SPLIT_RANK.get(split, len(SPLIT_PREFERENCE))  # unlisted names after the listed ones
        ranked.append((has_completions, preference, split))
    return [split for _, _, split in sorted(ranked)]


def _safe_rows_in(repo: str, dataset_dict) -> tuple[list[dict], dict] | None:
    """Safe prompts from the first usable split of a loaded DatasetDict (see _splits_in_preference_order)."""
    for split in _splits_in_preference_order(dataset_dict):
        columns = dataset_dict[split].column_names
        has_label_or_type = bool({"label", "type"} & set(columns))
        if "prompt" not in columns or not has_label_or_type:
            continue
        safe = pick_safe([dict(row) for row in dataset_dict[split]])
        if not safe:
            continue
        label_column = "label" if "label" in columns else "type (non-contrast)"
        source = {
            "repo": repo,
            "split": split,
            "columns": columns,
            "n_rows": len(dataset_dict[split]),
            "label_column": label_column,
        }
        return safe, source
    return None


def load_from_hub(repos: list[str]) -> tuple[list[dict], dict]:
    """Safe prompts and their source from the first repo that loads and has a usable split."""
    from datasets import load_dataset

    errors = []
    for repo in repos:
        try:
            dataset_dict = load_dataset(repo)
        except Exception as error:  # try the next repo
            errors.append(f"{repo}: {type(error).__name__}: {str(error)[:MAX_ERROR_CHARS]}")
            continue
        found = _safe_rows_in(repo, dataset_dict)
        if found:
            return found
        columns_by_split = {split: dataset_dict[split].column_names for split in dataset_dict}
        errors.append(f"{repo}: no split with a prompt column and safe rows ({columns_by_split})")
    raise RuntimeError("could not load XSTest:\n  " + "\n  ".join(errors))


def load_local(path: Path) -> list[dict]:
    """Rows of a local XSTest CSV or JSONL file."""
    if path.suffix.lower() == ".csv":
        with open(path, newline="", encoding="utf-8-sig") as file:
            return list(csv.DictReader(file))
    return read_jsonl(path)


def load_safe_prompts(args, config: dict) -> tuple[list[dict], dict]:
    """(safe XSTest rows, source record) from --input-file or the Hub."""
    if args.input_file:
        path = Path(args.input_file)
        return pick_safe(load_local(path)), {"repo": f"local:{path.name}", "split": "file"}
    repos = hub_candidates(args.source, config["datasets"].get("xstest"), *CANDIDATES)
    return load_from_hub(repos)


# ── English rows and splits ──────────────────────────────────────────────────


def make_rows(safe: list[dict], source: dict) -> list[dict]:
    """XSTest rows as our {"id", "prompt", "metadata"} English rows."""
    rows = []
    for raw in safe:
        prompt = normalize_text(str(raw["prompt"]))
        metadata = {
            "lang": "en",
            "language": "English",
            "source": source["repo"],
            "source_split": source["split"],
            "label": "safe",
            "xstest_type": raw.get("type"),
            "focus": raw.get("focus"),
            "note": raw.get("note"),
        }
        if raw.get("id") not in (None, ""):
            metadata["xstest_id"] = raw["id"]
        rows.append({"id": stable_id("en", prompt), "prompt": prompt, "metadata": metadata})
    return rows


def _with_split(row: dict, split: str) -> dict:
    """A copy of the row whose metadata also records its split."""
    metadata = dict(row["metadata"])
    metadata["split"] = split
    return dict(row, metadata=metadata)


def write_english_splits(by_id: dict[str, dict], fractions: dict, seed: int, out_dir: Path) -> dict[str, list[str]]:
    """Split the English rows (stratified by XSTest type), write one file per split; returns the split ids."""
    strata = [row["metadata"]["xstest_type"] for row in by_id.values()]
    parts = split_ids(list(by_id), fractions, seed, strata=strata)
    for split, ids in parts.items():
        write_jsonl(out_dir / f"{split}.jsonl", [_with_split(by_id[row_id], split) for row_id in ids])
    counts = ", ".join(f"{split}={len(ids)}" for split, ids in parts.items())
    print(f"[xstest] en: {counts} -> {out_dir}")
    return parts


# ── main ─────────────────────────────────────────────────────────────────────


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--data-dir",
        default=None,
        help="processed data root to write (default: data/processed; outputs/dry_run/processed with --dry-run)",
    )
    parser.add_argument("--source", default=None, help="Hub id to try first")
    parser.add_argument(
        "--input-file", default=None, help="local XSTest CSV/JSONL (prompt, type, label, ...) instead of the Hub"
    )
    parser.add_argument("--dev-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=None, help="default: config split_seed")
    parser.add_argument("--skip-llm", action="store_true", help="only write the English files")
    parser.add_argument("--log-raw-rejects", action="store_true", help="keep 300-char previews of rejected outputs")
    parser.add_argument("--targets", nargs="+", choices=["hi", "hinglish"], default=["hi", "hinglish"])
    make_hinglish.add_llm_args(parser)
    return parser.parse_args(argv)


def output_root(args) -> Path:
    """--data-dir, else the dry-run folder with --dry-run, else data/processed."""
    if args.data_dir:
        return resolve_data_dir(args.data_dir)
    if args.dry_run:
        return resolve_data_dir(make_hinglish.DRY_RUN_DIR)
    return resolve_data_dir(None)


def warn_if_unexpected_count(safe: list[dict]) -> None:
    """Print a warning, with the per-type counts, unless there are XSTest v2's 250 safe prompts."""
    if len(safe) == EXPECTED_SAFE:
        return
    type_counts = dict(Counter(str(row.get("type")) for row in safe))
    print(f"[xstest] WARNING: expected {EXPECTED_SAFE} safe prompts (XSTest v2), got {len(safe)}; types: {type_counts}")


def convert(args, target: str, llm, data_dir: Path, seed: int) -> dict:
    """Translate (hi) or convert (hinglish) the English splits with 01_make_hinglish.py; returns its summary."""
    return make_hinglish.run_conversion(
        set_name="xstest",
        target=target,
        llm=llm,
        data_dir=data_dir,
        out_dir=data_dir,
        sources=["en"],
        review_dir=Path(args.review_dir),
        review_frac=args.review_frac,
        seed=seed,
        ratio_bounds=(args.ratio_min, args.ratio_max),
        require_hinglish=not args.no_require_hinglish,
        concurrency=args.concurrency,
        max_parse_retries=args.max_parse_retries,
        log_raw=args.log_raw_rejects,
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    seed = config["split_seed"] if args.seed is None else args.seed
    data_dir = output_root(args)

    safe, source = load_safe_prompts(args, config)
    print(f"[xstest] source used: {source['repo']} (split {source['split']!r}); {len(safe)} safe prompts")
    warn_if_unexpected_count(safe)

    by_id = {row["id"]: row for row in make_rows(safe, source)}
    if len(by_id) != len(safe):
        raise RuntimeError("two XSTest prompts share an id after normalisation")
    fractions = {"dev": args.dev_frac, "test": 1.0 - args.dev_frac}
    parts = write_english_splits(by_id, fractions, seed, data_dir / "xstest" / "en")

    type_counts = {}
    for split, ids in parts.items():
        type_counts[split] = dict(Counter(by_id[row_id]["metadata"]["xstest_type"] for row_id in ids))
    manifest = {
        "name": "xstest",
        "built": timestamp(),
        "git": git_hash(),
        "source": source,
        "seed": seed,
        "fractions": fractions,
        "n_safe": len(safe),
        "type_counts": type_counts,
        "counts": {"en": {split: len(ids) for split, ids in parts.items()}},
        "conversions": {},
    }
    if not args.skip_llm:
        llm = make_hinglish.make_llm(
            args.model, args.api_base, dry_run=args.dry_run, temperature=args.temperature, max_tokens=args.max_tokens
        )
        for target in args.targets:
            summary = convert(args, target, llm, data_dir, seed)
            manifest["conversions"][target] = summary
            manifest["counts"][target] = {split: info["n_out"] for split, info in summary["splits"].items()}
    manifest_path = write_manifest(data_dir / "xstest" / "manifest.json", manifest)
    print(f"[xstest] manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
