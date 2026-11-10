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
EXPECTED_SAFE = 250


def _load_hinglish_module():
    """scripts/01_make_hinglish.py, whose file name cannot be imported normally."""
    path = Path(__file__).resolve().with_name("01_make_hinglish.py")
    spec = importlib.util.spec_from_file_location("make_hinglish", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


make_hinglish = _load_hinglish_module()


def is_safe(row: dict) -> bool:
    label = str(row.get("label") or "").strip().lower()
    if label in ("safe", "unsafe"):
        return label == "safe"
    typ = str(row.get("type") or "").strip().lower()
    return bool(typ) and not typ.startswith("contrast")


def pick_safe(rows: list[dict]) -> list[dict]:
    """Safe rows, deduplicated by normalised prompt (keeps the first)."""
    out, seen = [], set()
    for r in rows:
        key = normalize_for_match(str(r.get("prompt") or ""))
        if is_safe(r) and key and key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _safe_rows_in(repo: str, dd) -> tuple[list[dict], dict] | None:
    """Safe prompts from the first usable split of a loaded DatasetDict: splits without model completions
    first, then test, train, prompts, then the rest by name."""

    def rank(split: str):
        has_completions = any(c in dd[split].column_names for c in ("completion", "response"))
        return has_completions, SPLIT_PREFERENCE.index(split) if split in SPLIT_PREFERENCE else 99, split

    for split in sorted(dd, key=rank):
        cols = dd[split].column_names
        if "prompt" not in cols or not ({"label", "type"} & set(cols)):
            continue
        safe = pick_safe([dict(r) for r in dd[split]])
        if safe:
            return safe, {
                "repo": repo,
                "split": split,
                "columns": cols,
                "n_rows": len(dd[split]),
                "label_column": "label" if "label" in cols else "type (non-contrast)",
            }
    return None


def load_from_hub(repos: list[str]) -> tuple[list[dict], dict]:
    from datasets import load_dataset

    errors = []
    for repo in repos:
        try:
            dd = load_dataset(repo)
        except Exception as e:  # try the next repo
            errors.append(f"{repo}: {type(e).__name__}: {str(e)[:200]}")
            continue
        found = _safe_rows_in(repo, dd)
        if found:
            return found
        errors.append(f"{repo}: no split with a prompt column and safe rows ({ {s: dd[s].column_names for s in dd} })")
    raise RuntimeError("could not load XSTest:\n  " + "\n  ".join(errors))


def load_local(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with open(path, newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    return read_jsonl(path)


def make_rows(safe: list[dict], source: dict) -> list[dict]:
    rows = []
    for r in safe:
        prompt = normalize_text(str(r["prompt"]))
        meta = {
            "lang": "en",
            "language": "English",
            "source": source["repo"],
            "source_split": source["split"],
            "label": "safe",
            "xstest_type": r.get("type"),
            "focus": r.get("focus"),
            "note": r.get("note"),
        }
        if r.get("id") not in (None, ""):
            meta["xstest_id"] = r["id"]
        rows.append({"id": stable_id("en", prompt), "prompt": prompt, "metadata": meta})
    return rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__
    )
    p.add_argument("--config", default=None)
    p.add_argument(
        "--data-dir",
        default=None,
        help="processed data root to write (default: data/processed; outputs/dry_run/processed with --dry-run)",
    )
    p.add_argument("--source", default=None, help="Hub id to try first")
    p.add_argument(
        "--input-file", default=None, help="local XSTest CSV/JSONL (prompt, type, label, ...) instead of the Hub"
    )
    p.add_argument("--dev-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=None, help="default: config split_seed")
    p.add_argument("--skip-llm", action="store_true", help="only write the English files")
    p.add_argument("--log-raw-rejects", action="store_true", help="keep 300-char previews of rejected outputs")
    p.add_argument("--targets", nargs="+", choices=["hi", "hinglish"], default=["hi", "hinglish"])
    make_hinglish.add_llm_args(p)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    seed = cfg["split_seed"] if args.seed is None else args.seed
    data_dir = resolve_data_dir(args.data_dir or (make_hinglish.DRY_RUN_DIR if args.dry_run else None))

    if args.input_file:
        path = Path(args.input_file)
        safe = pick_safe(load_local(path))
        source = {"repo": f"local:{path.name}", "split": "file"}
    else:
        safe, source = load_from_hub(hub_candidates(args.source, cfg["datasets"].get("xstest"), *CANDIDATES))
    print(f"[xstest] source used: {source['repo']} (split {source['split']!r}); {len(safe)} safe prompts")
    if len(safe) != EXPECTED_SAFE:
        print(
            f"[xstest] WARNING: expected {EXPECTED_SAFE} safe prompts (XSTest v2), got {len(safe)}; "
            f"types: {dict(Counter(str(r.get('type')) for r in safe))}"
        )

    by_id = {r["id"]: r for r in make_rows(safe, source)}
    if len(by_id) != len(safe):
        raise RuntimeError("two XSTest prompts share an id after normalisation")
    fractions = {"dev": args.dev_frac, "test": 1.0 - args.dev_frac}
    parts = split_ids(list(by_id), fractions, seed, strata=[r["metadata"]["xstest_type"] for r in by_id.values()])
    for split, ids in parts.items():
        split_rows = [dict(by_id[i], metadata={**by_id[i]["metadata"], "split": split}) for i in ids]
        write_jsonl(data_dir / "xstest" / "en" / f"{split}.jsonl", split_rows)
    print("[xstest] en: " + ", ".join(f"{s}={len(v)}" for s, v in parts.items()) + f" -> {data_dir / 'xstest' / 'en'}")

    manifest = {
        "name": "xstest",
        "built": timestamp(),
        "git": git_hash(),
        "source": source,
        "seed": seed,
        "fractions": fractions,
        "n_safe": len(safe),
        "type_counts": {s: dict(Counter(by_id[i]["metadata"]["xstest_type"] for i in ids)) for s, ids in parts.items()},
        "counts": {"en": {s: len(v) for s, v in parts.items()}},
        "conversions": {},
    }
    if not args.skip_llm:
        llm = make_hinglish.make_llm(
            args.model, args.api_base, dry_run=args.dry_run, temperature=args.temperature, max_tokens=args.max_tokens
        )
        for target in args.targets:
            summary = make_hinglish.run_conversion(
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
            manifest["conversions"][target] = summary
            manifest["counts"][target] = {s: v["n_out"] for s, v in summary["splits"].items()}
    mpath = write_manifest(data_dir / "xstest" / "manifest.json", manifest)
    print(f"[xstest] manifest -> {mpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
