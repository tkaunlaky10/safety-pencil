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
from collections import Counter
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

MAX_IDS_LISTED = 50  # shared ids listed per pair of sets
CROSS_LINGUAL_THRESHOLD = 0.8  # near threshold for the cross-lingual info when the near check is off
REPORTED_ROLES = ("test", "factor", "train", "dev")


# ── finding the sets ─────────────────────────────────────────────────────────


def discover(data_dir: Path) -> dict[str, list[dict]]:
    """{"harmful/en/test": rows, "task_t1/heldout": rows, ...} for every processed file that exists."""
    sets: dict[str, list[dict]] = {}
    for name, splits in PROMPT_SET_SPLITS.items():
        for lang in LANGS:
            for split in splits:
                path = split_path(data_dir, name, split, lang)
                if path.exists():
                    sets[f"{name}/{lang}/{split}"] = read_jsonl(path)
    task_dir = data_dir / "task_t1"
    for path in sorted(task_dir.glob("train_*.jsonl")) + [task_dir / "heldout.jsonl"]:
        if path.exists():
            sets[f"task_t1/{path.stem}"] = read_jsonl(path)
    for path in sorted((data_dir / "contamination").glob("beavertails_*.jsonl")):
        sets[f"contamination/{path.stem}"] = read_jsonl(path)
    return sets


def role(name: str) -> str:
    """What a set is used for: "test", "factor", "train", "dev" or "other"."""
    if name == "task_t1/heldout" or name.endswith("/test"):
        return "test"
    if name.endswith("/factor"):
        return "factor"
    if name.startswith(("task_t1/train_", "contamination/")):
        return "train"
    if name.endswith("/dev"):
        return "dev"
    return "other"


# ── checks ───────────────────────────────────────────────────────────────────


def id_collisions(pairs: list[tuple[str, str]], ids: dict[str, list[str]]) -> list[dict]:
    """Pairs of sets that share ids (up to 50 listed per pair)."""
    collisions = []
    for a, b in pairs:
        shared = sorted(set(ids[a]) & set(ids[b]))
        if shared:
            collisions.append({"a": a, "b": b, "n": len(shared), "ids": shared[:MAX_IDS_LISTED]})
    return collisions


def _split_of(set_name: str) -> str:
    """The split part of a set name: "harmful/hi/dev:literal_translation" -> "dev"."""
    return set_name.split("/")[2].split(":")[0]


def _file_set_name(set_name: str) -> str:
    """The set name without a ":field" suffix, i.e. the file the texts came from."""
    return set_name.split(":")[0]


def cross_lingual_info(
    sets: dict[str, list[dict]], texts: dict[str, list[str]], ids: dict[str, list[str]], near: float | None
) -> dict | None:
    """Overlap of the English literal translations of Hindi harmful prompts with the English harmful prompts
    of the other splits; None when no Hindi file has literal translations."""
    literal = {}
    for name, rows in sets.items():
        if not name.startswith("harmful/hi/"):
            continue
        translations = [str(row.get("metadata", {}).get("literal_translation") or "") for row in rows]
        if any(translations):
            literal[f"{name}:literal_translation"] = translations
    if not literal:
        return None

    english = [name for name in sets if name.startswith("harmful/en/")]
    pairs = [(a, b) for a in literal for b in english if _split_of(a) != _split_of(b)]
    compared = {**literal, **{name: texts[name] for name in english}}
    compared_ids = {name: ids[_file_set_name(name)] for name in compared}
    threshold = near if near else CROSS_LINGUAL_THRESHOLD
    return check_no_overlap(compared, pairs=pairs, near_threshold=threshold, ids=compared_ids)


def print_hits(report: dict, label: str, near_fails: bool) -> None:
    """One line per pair with matches, tagged ``label`` if it fails and "near" if only near matches warn."""
    hits = [pair for pair in report["pairs"] if pair["n_exact"] or pair.get("n_near")]
    for pair in hits:
        near_text = ""
        if pair.get("n_near") is not None:
            near_text = f", {pair['n_near']} near"
        tag = label if (pair["n_exact"] or near_fails) else "near"
        print(f"  [{tag}] {pair['a']} vs {pair['b']}: {pair['n_exact']} exact{near_text}")
    if not hits:
        print("  none")


def n_failing(report: dict, fail_on_near: bool) -> int:
    """Matches that fail the check: exact ones, plus near ones with --fail-on-near."""
    if not fail_on_near:
        return report["n_exact"]
    n_near = report["n_near"] or 0  # None when the near check is off
    return report["n_exact"] + n_near


# ── main ─────────────────────────────────────────────────────────────────────


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__
    )
    parser.add_argument("--data-dir", default=None, help="processed data root (default: data/processed)")
    parser.add_argument(
        "--near-threshold",
        type=float,
        default=0.8,
        help="character 5-gram Jaccard for near duplicates (<= 0 turns the near check off)",
    )
    parser.add_argument(
        "--fail-on-near", action="store_true", help="near duplicates in test-vs-factor/train pairs fail"
    )
    parser.add_argument("--strict", action="store_true", help="warnings (test-dev, dev-factor/train) fail too")
    parser.add_argument("--out-dir", default=None, help="results folder (default: outputs/results)")
    parser.add_argument("--name", default="data_contamination_check", help="result name (JSON file stem)")
    return parser.parse_args(argv)


def print_inventory(sets: dict, roles: dict[str, str], data_dir: Path) -> None:
    """How many files were found, by role."""
    role_counts = Counter(roles.values())
    by_role = ", ".join(f"{name}={role_counts[name]}" for name in REPORTED_ROLES)
    print(f"[contamination] {len(sets)} files under {data_dir}: {by_role}")


def main(argv=None) -> int:
    args = parse_args(argv)
    data_dir = resolve_data_dir(args.data_dir)
    sets = discover(data_dir)
    roles = {name: role(name) for name in sets}
    tests = [name for name in sets if roles[name] == "test"]
    fit = [name for name in sets if roles[name] in ("factor", "train")]
    devs = [name for name in sets if roles[name] == "dev"]
    print_inventory(sets, roles, data_dir)
    if not tests or not fit:
        print("[contamination] nothing to compare: need at least one test file and one factor/train file")
        return 2

    near = args.near_threshold if args.near_threshold > 0 else None
    texts = {name: [row.get("prompt", "") for row in rows] for name, rows in sets.items()}
    ids = {name: [str(row.get("id")) for row in rows] for name, rows in sets.items()}
    fail_pairs = [(test, fit_set) for test in tests for fit_set in fit]
    warn_pairs = [(test, dev) for test in tests for dev in devs] + [(dev, fit_set) for dev in devs for fit_set in fit]
    fail = check_no_overlap(texts, pairs=fail_pairs, near_threshold=near, ids=ids)
    warn = None
    if warn_pairs:
        warn = check_no_overlap(texts, pairs=warn_pairs, near_threshold=near, ids=ids)
    collisions = id_collisions(fail_pairs, ids)
    info = cross_lingual_info(sets, texts, ids, near)

    print("[contamination] test vs factor/train:")
    print_hits(fail, "FAIL", args.fail_on_near)
    n_fail = n_failing(fail, args.fail_on_near) + sum(collision["n"] for collision in collisions)
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
        "sizes": {name: len(rows) for name, rows in sets.items()},
        "roles": roles,
        "passed": n_fail == 0,
        "n_fail": n_fail,
        "test_vs_fit": fail,
        "id_collisions": collisions,
        "warnings": warn,
        "cross_lingual_info": info,
    }
    config = {"near_threshold": near, "fail_on_near": args.fail_on_near, "strict": args.strict}
    path = write_result(args.name, payload, config=config, out_dir=args.out_dir)
    status = "PASS" if n_fail == 0 else f"FAIL ({n_fail})"
    print(f"[contamination] {status}; report -> {path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
