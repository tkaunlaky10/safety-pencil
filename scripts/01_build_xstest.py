#!/usr/bin/env python
"""Build the over-refusal set: XSTest's safe prompts in en, plus hi and hinglish via an LLM.

1. Load XSTest from the first source that works (``--source`` first, then config datasets.xstest,
   walledai/XSTest, natolambert/xstest-v2-copy, Paul/XSTest) and keep the safe prompts: label
   "safe", or, without a label column, a type not starting with "contrast". The source used is
   printed and recorded. Official XSTest v2 has 250 safe prompts, 25 for each of 10 types; a
   different count is printed as a warning.
2. Split dev/test 20/80 with ``split_ids`` (seeded, stratified by XSTest type) and write
   ``xstest/en/{dev,test}.jsonl``.
3. Unless ``--skip-llm``: translate to Hindi (``HindiTranslationTask``) and convert to Hinglish
   (``HinglishTask``), both from the English rows, through the same CuratorKIT path as
   01_make_hinglish.py: ``xstest/hi/{split}.jsonl`` (ids ``<en_id>-hi``) and
   ``xstest/hinglish/{split}.jsonl`` (ids ``<en_id>-hinglish``). Review samples go to
   outputs/review/{hi,hinglish}_check_xstest.csv.

``--dry-run`` uses the fake LLM and writes under outputs/dry_run/processed unless --data-dir is set.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pencil.config import load_config  # noqa: E402
from pencil.datasets import (  # noqa: E402
    REPO_ROOT, normalize_for_match, normalize_text, read_jsonl, resolve_data_dir, split_ids,
    stable_id, write_jsonl,
)
from pencil.results import git_hash  # noqa: E402

CANDIDATES = ("walledai/XSTest", "natolambert/xstest-v2-copy", "Paul/XSTest")
SPLIT_PREFERENCE = ("test", "train", "prompts")
EXPECTED_SAFE = 250


def _load_hinglish_module():
    path = Path(__file__).resolve().with_name("01_make_hinglish.py")
    spec = importlib.util.spec_from_file_location("make_hinglish", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def is_safe(row: dict) -> bool | None:
    label = str(row.get("label") or "").strip().lower()
    if label in ("safe", "unsafe"):
        return label == "safe"
    typ = str(row.get("type") or "").strip().lower()
    if typ:
        return not typ.startswith("contrast")
    return None


def pick_safe(rows: list[dict]) -> list[dict]:
    """Safe rows, deduplicated by normalised prompt (keeps the first)."""
    out, seen = [], set()
    for r in rows:
        prompt = str(r.get("prompt") or "").strip()
        key = normalize_for_match(prompt)
        if is_safe(r) and key and key not in seen:
            seen.add(key)
            out.append(r)
    return out


def load_from_hub(repos: list[str]) -> tuple[list[dict], dict]:
    from datasets import load_dataset

    errors = []
    for repo in repos:
        try:
            dd = load_dataset(repo)
        except Exception as e:
            errors.append(f"{repo}: {type(e).__name__}: {str(e)[:200]}")
            continue
        names = sorted(dd, key=lambda s: (SPLIT_PREFERENCE.index(s) if s in SPLIT_PREFERENCE else 99, s))
        # prefer splits that hold prompts only, not model completions
        names.sort(key=lambda s: any(c in dd[s].column_names for c in ("completion", "response")))
        for split in names:
            cols = dd[split].column_names
            if "prompt" not in cols or not ({"label", "type"} & set(cols)):
                continue
            safe = pick_safe([dict(r) for r in dd[split]])
            if safe:
                info = {"repo": repo, "split": split, "columns": cols, "n_rows": len(dd[split]),
                        "label_column": "label" if "label" in cols else "type (non-contrast)"}
                return safe, info
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
        meta = {"lang": "en", "language": "English", "source": source["repo"], "source_split": source["split"],
                "label": "safe", "xstest_type": r.get("type"), "focus": r.get("focus"), "note": r.get("note")}
        if r.get("id") not in (None, ""):
            meta["xstest_id"] = r["id"]
        rows.append({"id": stable_id("en", prompt), "prompt": prompt, "metadata": meta})
    return rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--data-dir", default=None,
                   help="processed data root to write (default: data/processed; outputs/dry_run/processed with --dry-run)")
    p.add_argument("--source", default=None, help="Hub id to try first")
    p.add_argument("--input-file", default=None, help="local XSTest CSV/JSONL (prompt, type, label, ...) instead of the Hub")
    p.add_argument("--dev-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=None, help="default: config split_seed")
    p.add_argument("--skip-llm", action="store_true", help="only write the English files")
    p.add_argument("--log-raw-rejects", action="store_true", help="keep 300-char previews of rejected outputs")
    p.add_argument("--targets", nargs="+", choices=["hi", "hinglish"], default=["hi", "hinglish"])
    hm = _load_hinglish_module()
    hm.add_llm_args(p)
    return p.parse_args(argv), hm


def main(argv=None) -> int:
    args, hm = parse_args(argv)
    cfg = load_config(args.config)
    seed = cfg["split_seed"] if args.seed is None else args.seed
    if args.data_dir:
        data_dir = resolve_data_dir(args.data_dir)
    else:
        data_dir = REPO_ROOT / "outputs" / "dry_run" / "processed" if args.dry_run else resolve_data_dir(None)

    if args.input_file:
        path = Path(args.input_file)
        safe = pick_safe(load_local(path))
        source = {"repo": f"local:{path.name}", "split": "file"}
    else:
        repos = list(dict.fromkeys([r for r in (args.source, cfg["datasets"].get("xstest")) if r] + list(CANDIDATES)))
        safe, source = load_from_hub(repos)
    print(f"[xstest] source used: {source['repo']} (split {source['split']!r}); {len(safe)} safe prompts")
    types = Counter(str(r.get("type")) for r in safe)
    if len(safe) != EXPECTED_SAFE:
        print(f"[xstest] WARNING: expected {EXPECTED_SAFE} safe prompts (XSTest v2), got {len(safe)}; "
              f"types: {dict(types)}")

    rows = make_rows(safe, source)
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise RuntimeError("two XSTest prompts share an id after normalisation")
    fractions = {"dev": args.dev_frac, "test": 1.0 - args.dev_frac}
    parts = split_ids(list(by_id), fractions, seed, strata=[by_id[i]["metadata"]["xstest_type"] for i in by_id])
    for split, ids in parts.items():
        split_rows = [dict(by_id[i], metadata={**by_id[i]["metadata"], "split": split}) for i in ids]
        write_jsonl(data_dir / "xstest" / "en" / f"{split}.jsonl", split_rows)
    print("[xstest] en: " + ", ".join(f"{s}={len(v)}" for s, v in parts.items()) + f" -> {data_dir / 'xstest' / 'en'}")

    manifest = {"name": "xstest", "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git": git_hash(),
                "source": source, "seed": seed, "fractions": fractions, "n_safe": len(safe),
                "type_counts": {s: dict(Counter(by_id[i]["metadata"]["xstest_type"] for i in ids))
                                for s, ids in parts.items()},
                "counts": {"en": {s: len(v) for s, v in parts.items()}}, "conversions": {}}
    if not args.skip_llm:
        llm = hm.make_llm(args.model, args.api_base, dry_run=args.dry_run, temperature=args.temperature,
                          max_tokens=args.max_tokens)
        for target in args.targets:
            summary = hm.run_conversion(
                set_name="xstest", target=target, llm=llm, data_dir=data_dir, out_dir=data_dir,
                sources=["en"], review_dir=Path(args.review_dir), review_frac=args.review_frac, seed=seed,
                ratio_bounds=(args.ratio_min, args.ratio_max), require_hinglish=not args.no_require_hinglish,
                concurrency=args.concurrency, max_parse_retries=args.max_parse_retries, log_raw=args.log_raw_rejects)
            manifest["conversions"][target] = summary
            manifest["counts"][target] = {s: v["n_out"] for s, v in summary["splits"].items()}
    mpath = data_dir / "xstest" / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str))
    print(f"[xstest] manifest -> {mpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
