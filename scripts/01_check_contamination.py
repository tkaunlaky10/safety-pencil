#!/usr/bin/env python
"""Check that no test prompt leaks into a factor or training file.

Reads whatever exists of the processed layout (see ``pencil.datasets``) and compares prompts after
``normalize_for_match`` (NFKC, casefold, punctuation removed):

- FAIL (exit 1): a test set (harmful/*/test, xstest/*/test, task_t1/heldout) sharing a prompt or an id with
  a factor set (harmful/*/factor) or a training set (task_t1/train_*, contamination/*). With
  --fail-on-near, near duplicates in these pairs fail too.
- WARN: test vs dev, and dev vs factor/train (dev picks mu and beta, so overlaps there matter less but are
  still reported). With --strict, these fail as well.
- INFO: English literal translations of Hindi harmful prompts vs English prompts of the other splits
  (cross-lingual near-twins; the splits are per language, so this never fails).

Near duplicates: character 5-gram Jaccard >= --near-threshold on the normalised text. The JSON report (ids
and indices, no prompt text) goes through ``pencil.results.write_result`` to
outputs/results/data_contamination_check.json. Exit 2 if there is nothing to compare.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pencil.datasets import (
    LANGS,
    PROMPT_SET_SPLITS,
    check_no_overlap,
    read_jsonl,
    resolve_data_dir,
    split_path,
)
from pencil.results import write_result


def discover(data_dir: Path) -> dict[str, list[dict]]:
    """{"harmful/en/test": rows, "task_t1/heldout": rows, ...} for every processed file that exists."""
    sets: dict[str, list[dict]] = {}
    for name, splits in PROMPT_SET_SPLITS.items():
        for lang in LANGS:
            for split in splits:
                p = split_path(data_dir, name, split, lang)
                if p.exists():
                    sets[f"{name}/{lang}/{split}"] = read_jsonl(p)
    t1 = data_dir / "task_t1"
    for p in sorted(t1.glob("train_*.jsonl")) + [t1 / "heldout.jsonl"]:
        if p.exists():
            sets[f"task_t1/{p.stem}"] = read_jsonl(p)
    for p in sorted((data_dir / "contamination").glob("beavertails_*.jsonl")):
        sets[f"contamination/{p.stem}"] = read_jsonl(p)
    return sets


def role(name: str) -> str:
    if name == "task_t1/heldout" or name.endswith("/test"):
        return "test"
    if name.endswith("/factor"):
        return "factor"
    if name.startswith(("task_t1/train_", "contamination/")):
        return "train"
    if name.endswith("/dev"):
        return "dev"
    return "other"


def id_collisions(pairs: list[tuple[str, str]], ids: dict[str, list[str]]) -> list[dict]:
    """Pairs of sets that share ids (up to 50 listed per pair)."""
    out = []
    for a, b in pairs:
        shared = sorted(set(ids[a]) & set(ids[b]))
        if shared:
            out.append({"a": a, "b": b, "n": len(shared), "ids": shared[:50]})
    return out


def cross_lingual_info(
    sets: dict[str, list[dict]], texts: dict[str, list[str]], ids: dict[str, list[str]], near: float | None
) -> dict | None:
    """Overlap of the English literal translations of Hindi harmful prompts with the English harmful prompts
    of the other splits; None when no Hindi file has literal translations."""
    literal = {}
    for name, rows in sets.items():
        if name.startswith("harmful/hi/"):
            translations = [str(r.get("metadata", {}).get("literal_translation") or "") for r in rows]
            if any(translations):
                literal[f"{name}:literal_translation"] = translations
    if not literal:
        return None
    english = [k for k in sets if k.startswith("harmful/en/")]

    def split_of(name: str) -> str:
        return name.split("/")[2].split(":")[0]

    pairs = [(a, b) for a in literal for b in english if split_of(a) != split_of(b)]
    xsets = {**literal, **{k: texts[k] for k in english}}
    return check_no_overlap(
        xsets, pairs=pairs, near_threshold=near or 0.8, ids={k: ids[k.split(":")[0]] for k in xsets}
    )


def print_hits(report: dict, label: str, near_fails: bool) -> None:
    hits = [p for p in report["pairs"] if p["n_exact"] or p.get("n_near")]
    for p in hits:
        near = f", {p['n_near']} near" if p.get("n_near") is not None else ""
        tag = label if (p["n_exact"] or near_fails) else "near"
        print(f"  [{tag}] {p['a']} vs {p['b']}: {p['n_exact']} exact{near}")
    if not hits:
        print("  none")


def n_failing(report: dict, fail_on_near: bool) -> int:
    return report["n_exact"] + ((report["n_near"] or 0) if fail_on_near else 0)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__
    )
    p.add_argument("--data-dir", default=None, help="processed data root (default: data/processed)")
    p.add_argument(
        "--near-threshold",
        type=float,
        default=0.8,
        help="character 5-gram Jaccard for near duplicates (<= 0 turns the near check off)",
    )
    p.add_argument("--fail-on-near", action="store_true", help="near duplicates in test-vs-factor/train pairs fail")
    p.add_argument("--strict", action="store_true", help="warnings (test-dev, dev-factor/train) fail too")
    p.add_argument("--out-dir", default=None, help="results folder (default: outputs/results)")
    p.add_argument("--name", default="data_contamination_check", help="result name (JSON file stem)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    data_dir = resolve_data_dir(args.data_dir)
    sets = discover(data_dir)
    roles = {k: role(k) for k in sets}
    tests = [k for k in sets if roles[k] == "test"]
    fit = [k for k in sets if roles[k] in ("factor", "train")]
    devs = [k for k in sets if roles[k] == "dev"]
    print(
        f"[contamination] {len(sets)} files under {data_dir}: "
        + ", ".join(f"{r}={sum(v == r for v in roles.values())}" for r in ("test", "factor", "train", "dev"))
    )
    if not tests or not fit:
        print("[contamination] nothing to compare: need at least one test file and one factor/train file")
        return 2

    near = args.near_threshold if args.near_threshold > 0 else None
    texts = {k: [r.get("prompt", "") for r in v] for k, v in sets.items()}
    ids = {k: [str(r.get("id")) for r in v] for k, v in sets.items()}
    fail_pairs = [(t, f) for t in tests for f in fit]
    warn_pairs = [(t, d) for t in tests for d in devs] + [(d, f) for d in devs for f in fit]
    fail = check_no_overlap(texts, pairs=fail_pairs, near_threshold=near, ids=ids)
    warn = check_no_overlap(texts, pairs=warn_pairs, near_threshold=near, ids=ids) if warn_pairs else None
    collisions = id_collisions(fail_pairs, ids)
    info = cross_lingual_info(sets, texts, ids, near)

    print("[contamination] test vs factor/train:")
    print_hits(fail, "FAIL", args.fail_on_near)
    n_fail = n_failing(fail, args.fail_on_near) + sum(c["n"] for c in collisions)
    if warn:
        print("[contamination] test vs dev, dev vs factor/train:")
        print_hits(warn, "FAIL" if args.strict else "WARN", args.strict and args.fail_on_near)
        if args.strict:
            n_fail += n_failing(warn, args.fail_on_near)
    if info and (info["n_exact"] or info["n_near"]):
        print(
            f"[contamination] info: {info['n_exact']} exact / {info['n_near']} near cross-lingual twins "
            "(hi literal translation vs en prompt in another split)"
        )

    payload = {
        "data_dir": str(data_dir),
        "sizes": {k: len(v) for k, v in sets.items()},
        "roles": roles,
        "passed": n_fail == 0,
        "n_fail": n_fail,
        "test_vs_fit": fail,
        "id_collisions": collisions,
        "warnings": warn,
        "cross_lingual_info": info,
    }
    path = write_result(
        args.name,
        payload,
        config={"near_threshold": near, "fail_on_near": args.fail_on_near, "strict": args.strict},
        out_dir=args.out_dir,
    )
    print(f"[contamination] {'PASS' if n_fail == 0 else f'FAIL ({n_fail})'}; report -> {path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
