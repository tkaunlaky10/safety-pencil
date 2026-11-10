#!/usr/bin/env python
"""Build the harmful-prompt splits (en, hi) from aya_redteaming.

For each language: load the Hub split (or a local JSONL with --input-jsonl) and give every row a
deterministic id. By default CuratorKIT then cleans it (min_tokens=1, exact + MinHash dedup, toxicity
filter off), exports ``prompts_meta`` and splits it factor/dev/test with its seeded ``output_split``,
keeping its own files in ``harmful/{lang}/_curatorkit/``. ``--no-curatorkit`` does exact + character
5-gram near-dup removal in plain Python instead. ``--stratify`` (either path) splits with
``pencil.datasets.split_ids`` stratified by harm category, which ``output_split`` cannot do. The split
modes give different (each deterministic) splits: pick one before anyone computes factors.

Writes ``data/processed/harmful/{lang}/{factor,dev,test}.jsonl`` and ``harmful/manifest.json`` (layout:
``pencil.datasets``). Hinglish comes afterwards, from these files, with ``01_make_hinglish.py``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from functools import partial
from pathlib import Path

from pencil.config import load_config
from pencil.datasets import (
    REPO_ROOT,
    clean_prompt,
    find_near_duplicates,
    hub_candidates,
    normalize_for_match,
    read_jsonl,
    resolve_data_dir,
    split_ids,
    stable_id,
    timestamp,
    write_jsonl,
    write_manifest,
)
from pencil.results import git_hash

HF_SPLIT = {"en": "english", "hi": "hindi"}
FALLBACK_REPOS = ("CohereLabs/aya_redteaming", "CohereForAI/aya_redteaming")
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw" / "aya_redteaming"  # gitignored


# ── loading ──────────────────────────────────────────────────────────────────


def _hub_revision(repo: str) -> str | None:
    """The dataset's current commit, for the manifest; None when the Hub API is unreachable."""
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(repo).sha
    except Exception:
        return None


def load_from_hub(repos: list[str], split: str) -> tuple[list[dict], dict]:
    """Try each repo, first with the language as a split of the default config (the card's YAML),
    then as a config name (the card's code example)."""
    from datasets import load_dataset

    errors = []
    for repo in repos:
        for how in ("split", "config"):
            try:
                if how == "split":
                    ds = load_dataset(repo, split=split)
                else:
                    dd = load_dataset(repo, split)
                    ds = dd["train"] if "train" in dd else dd[next(iter(dd))]
            except Exception as e:  # try the next way
                errors.append(f"{repo} ({how}={split!r}): {type(e).__name__}: {str(e)[:200]}")
                continue
            rows = [dict(r) for r in ds]
            info = {
                "repo": repo,
                "split": split,
                "loaded_as": how,
                "n_raw": len(rows),
                "revision": _hub_revision(repo),
                "fingerprint": getattr(ds, "_fingerprint", None),
            }
            print(f"[harmful] loaded {repo} ({how}={split!r}): {len(rows)} rows")
            return rows, info
    raise RuntimeError("could not load aya_redteaming:\n  " + "\n  ".join(errors))


def parse_categories(value) -> list[str]:
    """aya_redteaming stores harm_category as a JSON-list string, e.g. '["Violence, ..."]'."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    s = str(value).strip()
    if s.startswith("["):
        try:
            return [str(v).strip() for v in json.loads(s) if str(v).strip()]
        except json.JSONDecodeError:
            pass
    return [s] if s else []


def make_row(raw: dict, lang: str, source: dict) -> dict:
    """Raw aya_redteaming row -> {"id", "prompt", "metadata"} (id from the raw prompt text)."""
    prompt = str(raw.get("prompt") or "")
    meta = {k: v for k, v in raw.items() if k not in ("prompt", "id")}
    meta["harm_category"] = parse_categories(raw.get("harm_category"))
    meta.update(lang=lang, source=source["repo"], source_split=source["split"])
    if source.get("revision"):
        meta["source_revision"] = source["revision"]
    return {"id": stable_id(lang, prompt), "prompt": prompt, "metadata": meta}


def to_curator_sample(raw: dict, *, lang: str, source: dict):
    """CuratorKIT preprocessing_fn: build the DataSample ourselves (deterministic id, metadata as-is)."""
    from curatorkit.schema import DataSample

    row = make_row(raw, lang, source)
    return DataSample(
        id=row["id"],
        source_uri=f"hf://datasets/{source['repo']}/{source['split']}",
        instruction=row["prompt"],
        task_type="prompt_only",
        metadata=row["metadata"],
    )


# ── the two build paths ──────────────────────────────────────────────────────


def build_with_curatorkit(
    raw_path: Path,
    out_dir: Path,
    *,
    lang: str,
    source: dict,
    fractions: dict,
    seed: int,
    stratify: bool,
    minhash_threshold: float,
) -> tuple[dict[str, list[dict]], dict]:
    from curatorkit.curator import Curator, CuratorConfig

    if out_dir.exists():
        shutil.rmtree(out_dir)
    cfg = CuratorConfig(
        dataset=str(raw_path),
        format="prompt_only",
        preprocessing_fn=partial(to_curator_sample, lang=lang, source=source),
        min_tokens=1,  # default 10 whitespace words drops short prompts
        max_tokens=4096,
        dedup="minhash",  # exact, then MinHash
        minhash_threshold=minhash_threshold,
        toxicity_gate=False,  # these are harmful on purpose
        clean_transforms={"strip_html": False},  # keep prompts verbatim apart from whitespace/NFC
        export_formats=["prompts_meta"],
        output_split=None if stratify else dict(fractions),
        output_split_seed=seed,
        output_dir=str(out_dir),
        name=f"aya_redteaming_{lang}",
        write_hf_readme=False,  # we write the dataset cards ourselves
    )
    result = Curator(cfg).run()
    if source["n_raw"] and not result.passed:
        raise RuntimeError(
            f"CuratorKIT passed 0 of {source['n_raw']} rows for {lang}; see "
            f"{out_dir}/rejected.jsonl (or rerun with --no-curatorkit)"
        )

    def read(path: Path) -> list[dict]:
        return [
            {"id": r["id"], "prompt": r["prompt"], "metadata": dict(r.get("metadata") or {})} for r in read_jsonl(path)
        ]

    if stratify:
        splits = _split_rows(read(out_dir / "prompts_meta.jsonl"), fractions, seed, stratify=True)
    else:
        splits = {s: read(out_dir / s / "prompts_meta.jsonl") for s in fractions}
    stats = {
        "curatorkit_stage_counts": result.stage_counts,
        "n_rejected": len(result.rejected),
        "n_after_dedup": len(result.passed),
    }
    return splits, stats


def build_plain(
    raw_rows: list[dict], *, lang: str, source: dict, fractions: dict, seed: int, stratify: bool, near_threshold: float
) -> tuple[dict[str, list[dict]], dict]:
    rows, seen, n_empty, n_exact = [], set(), 0, 0
    for raw in raw_rows:
        row = make_row(raw, lang, source)
        row["prompt"] = clean_prompt(row["prompt"])
        key = normalize_for_match(row["prompt"])
        if not key:
            n_empty += 1
        elif key in seen:
            n_exact += 1
        else:
            seen.add(key)
            rows.append(row)
    # drop the later text of each near-duplicate pair whose earlier text is kept
    drop: set[int] = set()
    for i, j, _ in find_near_duplicates([r["prompt"] for r in rows], near_threshold):
        if i not in drop:
            drop.add(j)
    rows = [r for i, r in enumerate(rows) if i not in drop]
    splits = _split_rows(rows, fractions, seed, stratify=stratify)
    return splits, {"n_empty": n_empty, "n_exact_dups": n_exact, "n_near_dups": len(drop), "n_after_dedup": len(rows)}


def _split_rows(rows: list[dict], fractions: dict, seed: int, *, stratify: bool) -> dict[str, list[dict]]:
    by_id = {}
    for r in rows:  # identical raw prompts share an id: keep the first
        by_id.setdefault(r["id"], r)
    ids = list(by_id)
    strata = [by_id[i]["metadata"].get("harm_category") for i in ids] if stratify else None
    parts = split_ids(ids, fractions, seed, strata=strata)
    return {s: [by_id[i] for i in parts[s]] for s in fractions}


# ── main ─────────────────────────────────────────────────────────────────────


def category_counts(rows: list[dict]) -> dict[str, int]:
    c = Counter()
    for r in rows:
        c.update(r["metadata"].get("harm_category") or ["(none)"])
    return dict(sorted(c.items()))


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__
    )
    p.add_argument("--config", default=None, help="experiment.yaml (default: configs/experiment.yaml)")
    p.add_argument("--data-dir", default=None, help="processed data root (default: data/processed)")
    p.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="where Hub rows are cached as JSONL")
    p.add_argument("--langs", nargs="+", default=["en", "hi"], choices=sorted(HF_SPLIT))
    p.add_argument(
        "--input-jsonl",
        action="append",
        default=[],
        metavar="LANG=PATH",
        help="use a local raw JSONL (aya_redteaming columns) for LANG instead of the Hub",
    )
    p.add_argument("--dataset", default=None, help="Hub id to try first (default: config datasets.harmful)")
    p.add_argument("--seed", type=int, default=None, help="split seed (default: config split_seed)")
    p.add_argument("--no-curatorkit", action="store_true", help="plain-Python path instead of CuratorKIT")
    p.add_argument(
        "--stratify",
        action="store_true",
        help="split with split_ids stratified by harm_category (CuratorKIT still dedups)",
    )
    p.add_argument("--minhash-threshold", type=float, default=0.85, help="CuratorKIT MinHash threshold")
    p.add_argument(
        "--near-threshold",
        type=float,
        default=0.85,
        help="character 5-gram Jaccard threshold for --no-curatorkit near-dup removal",
    )
    return p.parse_args(argv)


def load_raw(lang: str, local: dict[str, str], repos: list[str], raw_dir: Path) -> tuple[list[dict], Path, dict]:
    """(raw rows, a JSONL file holding them, source info) for one language, from --input-jsonl or the Hub."""
    hf_split = HF_SPLIT[lang]
    if lang in local:
        raw_path = Path(local[lang])
        raw_rows = read_jsonl(raw_path)
        return (
            raw_rows,
            raw_path,
            {"repo": f"local:{raw_path.name}", "split": hf_split, "loaded_as": "jsonl", "n_raw": len(raw_rows)},
        )
    raw_rows, source = load_from_hub(repos, hf_split)
    return raw_rows, write_jsonl(raw_dir / f"{hf_split}.jsonl", raw_rows), source


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    fractions = {k: float(v) for k, v in cfg["split_fractions"].items()}
    seed = cfg["split_seed"] if args.seed is None else args.seed
    data_dir = resolve_data_dir(args.data_dir)
    local = dict(s.split("=", 1) for s in args.input_jsonl)
    unknown = set(local) - set(HF_SPLIT)
    if unknown:
        raise SystemExit(f"--input-jsonl languages must be in {sorted(HF_SPLIT)}, got {sorted(unknown)}")
    repos = hub_candidates(args.dataset, cfg["datasets"].get("harmful"), *FALLBACK_REPOS)
    split_mode = (
        "split_ids_stratified" if args.stratify else "split_ids" if args.no_curatorkit else "curatorkit_output_split"
    )

    built = {}
    for lang in args.langs:
        raw_rows, raw_path, source = load_raw(lang, local, repos, Path(args.raw_dir))
        out_lang = data_dir / "harmful" / lang
        if args.no_curatorkit:
            splits, stats = build_plain(
                raw_rows,
                lang=lang,
                source=source,
                fractions=fractions,
                seed=seed,
                stratify=args.stratify,
                near_threshold=args.near_threshold,
            )
        else:
            splits, stats = build_with_curatorkit(
                raw_path,
                out_lang / "_curatorkit",
                lang=lang,
                source=source,
                fractions=fractions,
                seed=seed,
                stratify=args.stratify,
                minhash_threshold=args.minhash_threshold,
            )
        all_ids = [r["id"] for rows in splits.values() for r in rows]
        if len(all_ids) != len(set(all_ids)):
            raise RuntimeError(f"{lang}: an id appears in more than one split")
        for split, rows in splits.items():
            for r in rows:
                r["metadata"]["split"] = split
            write_jsonl(out_lang / f"{split}.jsonl", rows)
        built[lang] = {
            "built": timestamp(),
            "git": git_hash(),
            "curatorkit": not args.no_curatorkit,
            "split_mode": split_mode,
            "fractions": fractions,
            "seed": seed,
            "source": source,
            **stats,
            "counts": {s: len(r) for s, r in splits.items()},
            "harm_category_counts": {s: category_counts(r) for s, r in splits.items()},
        }
        print(f"[harmful] {lang}: " + ", ".join(f"{s}={len(r)}" for s, r in splits.items()) + f" -> {out_lang}")

    mpath = data_dir / "harmful" / "manifest.json"
    earlier = json.loads(mpath.read_text())["languages"] if mpath.exists() else {}  # keep other languages
    write_manifest(mpath, {"name": "harmful", "updated": timestamp(), "languages": {**earlier, **built}})
    print(f"[harmful] manifest -> {mpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
