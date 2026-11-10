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
MAX_ERROR_CHARS = 200  # of each load error, in the message listing every attempt


# ── loading ──────────────────────────────────────────────────────────────────


def _hub_revision(repo: str) -> str | None:
    """The dataset's current commit, for the manifest; None when the Hub API is unreachable."""
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(repo).sha
    except Exception:
        return None


def _load_language(load_dataset, repo: str, split: str, loaded_as: str):
    """One Hub language, either as a split of the default config or as a config name ("split"/"config")."""
    if loaded_as == "split":
        return load_dataset(repo, split=split)
    dataset_dict = load_dataset(repo, split)
    if "train" in dataset_dict:
        return dataset_dict["train"]
    first_split = next(iter(dataset_dict))
    return dataset_dict[first_split]


def load_from_hub(repos: list[str], split: str) -> tuple[list[dict], dict]:
    """Try each repo, first with the language as a split of the default config (the card's YAML),
    then as a config name (the card's code example)."""
    from datasets import load_dataset

    errors = []
    for repo in repos:
        for loaded_as in ("split", "config"):
            try:
                dataset = _load_language(load_dataset, repo, split, loaded_as)
            except Exception as error:  # try the next way
                message = str(error)[:MAX_ERROR_CHARS]
                errors.append(f"{repo} ({loaded_as}={split!r}): {type(error).__name__}: {message}")
                continue
            rows = [dict(row) for row in dataset]
            info = {
                "repo": repo,
                "split": split,
                "loaded_as": loaded_as,
                "n_raw": len(rows),
                "revision": _hub_revision(repo),
                "fingerprint": getattr(dataset, "_fingerprint", None),
            }
            print(f"[harmful] loaded {repo} ({loaded_as}={split!r}): {len(rows)} rows")
            return rows, info
    raise RuntimeError("could not load aya_redteaming:\n  " + "\n  ".join(errors))


def parse_categories(value) -> list[str]:
    """aya_redteaming stores harm_category as a JSON-list string, e.g. '["Violence, ..."]'."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return _non_empty_stripped(value)
    text = str(value).strip()
    if text.startswith("["):
        try:
            return _non_empty_stripped(json.loads(text))
        except json.JSONDecodeError:
            pass
    if text:
        return [text]
    return []


def _non_empty_stripped(values) -> list[str]:
    """Each value as a stripped string, dropping the empty ones."""
    stripped = [str(value).strip() for value in values]
    return [text for text in stripped if text]


def make_row(raw: dict, lang: str, source: dict) -> dict:
    """Raw aya_redteaming row -> {"id", "prompt", "metadata"} (id from the raw prompt text)."""
    prompt = str(raw.get("prompt") or "")
    metadata = {key: value for key, value in raw.items() if key not in ("prompt", "id")}
    metadata["harm_category"] = parse_categories(raw.get("harm_category"))
    metadata["lang"] = lang
    metadata["source"] = source["repo"]
    metadata["source_split"] = source["split"]
    if source.get("revision"):
        metadata["source_revision"] = source["revision"]
    return {"id": stable_id(lang, prompt), "prompt": prompt, "metadata": metadata}


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
    """Clean, dedup and (unless stratify) split one language with CuratorKIT; returns the splits and stats."""
    from curatorkit.curator import Curator

    if out_dir.exists():
        shutil.rmtree(out_dir)
    config = _curator_config(
        raw_path,
        out_dir,
        lang=lang,
        source=source,
        fractions=fractions,
        seed=seed,
        stratify=stratify,
        minhash_threshold=minhash_threshold,
    )
    result = Curator(config).run()
    if source["n_raw"] and not result.passed:
        raise RuntimeError(
            f"CuratorKIT passed 0 of {source['n_raw']} rows for {lang}; see "
            f"{out_dir}/rejected.jsonl (or rerun with --no-curatorkit)"
        )

    if stratify:
        rows = _read_prompts_meta(out_dir / "prompts_meta.jsonl")
        splits = _split_rows(rows, fractions, seed, stratify=True)
    else:
        splits = {split: _read_prompts_meta(out_dir / split / "prompts_meta.jsonl") for split in fractions}
    stats = {
        "curatorkit_stage_counts": result.stage_counts,
        "n_rejected": len(result.rejected),
        "n_after_dedup": len(result.passed),
    }
    return splits, stats


def _curator_config(
    raw_path: Path,
    out_dir: Path,
    *,
    lang: str,
    source: dict,
    fractions: dict,
    seed: int,
    stratify: bool,
    minhash_threshold: float,
):
    """CuratorKIT settings for one language of aya_redteaming."""
    from curatorkit.curator import CuratorConfig

    # with stratify, CuratorKIT only cleans and dedups; we split afterwards with split_ids
    output_split = None if stratify else dict(fractions)
    return CuratorConfig(
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
        output_split=output_split,
        output_split_seed=seed,
        output_dir=str(out_dir),
        name=f"aya_redteaming_{lang}",
        write_hf_readme=False,  # we write the dataset cards ourselves
    )


def _read_prompts_meta(path: Path) -> list[dict]:
    """A CuratorKIT ``prompts_meta`` export as our {"id", "prompt", "metadata"} rows."""
    rows = []
    for exported in read_jsonl(path):
        metadata = dict(exported.get("metadata") or {})
        rows.append({"id": exported["id"], "prompt": exported["prompt"], "metadata": metadata})
    return rows


def build_plain(
    raw_rows: list[dict], *, lang: str, source: dict, fractions: dict, seed: int, stratify: bool, near_threshold: float
) -> tuple[dict[str, list[dict]], dict]:
    """Exact and near-duplicate removal plus split_ids, in plain Python; returns the splits and stats."""
    rows, n_empty, n_exact = _drop_empty_and_exact_duplicates(raw_rows, lang, source)
    rows, n_near = _drop_near_duplicates(rows, near_threshold)
    splits = _split_rows(rows, fractions, seed, stratify=stratify)
    stats = {"n_empty": n_empty, "n_exact_dups": n_exact, "n_near_dups": n_near, "n_after_dedup": len(rows)}
    return splits, stats


def _drop_empty_and_exact_duplicates(raw_rows: list[dict], lang: str, source: dict) -> tuple[list[dict], int, int]:
    """Rows with a cleaned prompt, without empty prompts and without repeats (after normalize_for_match);
    also returns how many of each were dropped."""
    rows = []
    seen = set()
    n_empty = 0
    n_exact = 0
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
    return rows, n_empty, n_exact


def _drop_near_duplicates(rows: list[dict], threshold: float) -> tuple[list[dict], int]:
    """Drop the later row of each near-duplicate pair whose earlier row is kept; also returns the count."""
    dropped: set[int] = set()
    for i, j, _ in find_near_duplicates([row["prompt"] for row in rows], threshold):
        if i not in dropped:
            dropped.add(j)
    kept = [row for index, row in enumerate(rows) if index not in dropped]
    return kept, len(dropped)


def _split_rows(rows: list[dict], fractions: dict, seed: int, *, stratify: bool) -> dict[str, list[dict]]:
    """Split rows with split_ids, optionally stratified by harm category."""
    by_id = {}
    for row in rows:  # identical raw prompts share an id: keep the first
        by_id.setdefault(row["id"], row)
    ids = list(by_id)
    strata = None
    if stratify:
        strata = [by_id[row_id]["metadata"].get("harm_category") for row_id in ids]
    parts = split_ids(ids, fractions, seed, strata=strata)
    return {split: [by_id[row_id] for row_id in parts[split]] for split in fractions}


# ── main ─────────────────────────────────────────────────────────────────────


def category_counts(rows: list[dict]) -> dict[str, int]:
    """Rows per harm category ("(none)" for rows without one), sorted by category."""
    counts = Counter()
    for row in rows:
        categories = row["metadata"].get("harm_category")
        if not categories:
            categories = ["(none)"]
        counts.update(categories)
    return dict(sorted(counts.items()))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__
    )
    parser.add_argument("--config", default=None, help="experiment.yaml (default: configs/experiment.yaml)")
    parser.add_argument("--data-dir", default=None, help="processed data root (default: data/processed)")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="where Hub rows are cached as JSONL")
    parser.add_argument("--langs", nargs="+", default=["en", "hi"], choices=sorted(HF_SPLIT))
    parser.add_argument(
        "--input-jsonl",
        action="append",
        default=[],
        metavar="LANG=PATH",
        help="use a local raw JSONL (aya_redteaming columns) for LANG instead of the Hub",
    )
    parser.add_argument("--dataset", default=None, help="Hub id to try first (default: config datasets.harmful)")
    parser.add_argument("--seed", type=int, default=None, help="split seed (default: config split_seed)")
    parser.add_argument("--no-curatorkit", action="store_true", help="plain-Python path instead of CuratorKIT")
    parser.add_argument(
        "--stratify",
        action="store_true",
        help="split with split_ids stratified by harm_category (CuratorKIT still dedups)",
    )
    parser.add_argument("--minhash-threshold", type=float, default=0.85, help="CuratorKIT MinHash threshold")
    parser.add_argument(
        "--near-threshold",
        type=float,
        default=0.85,
        help="character 5-gram Jaccard threshold for --no-curatorkit near-dup removal",
    )
    return parser.parse_args(argv)


def parse_local_inputs(input_jsonl: list[str]) -> dict[str, str]:
    """{lang: path} from the --input-jsonl LANG=PATH values; exits on an unknown language."""
    local = dict(value.split("=", 1) for value in input_jsonl)
    unknown = set(local) - set(HF_SPLIT)
    if unknown:
        raise SystemExit(f"--input-jsonl languages must be in {sorted(HF_SPLIT)}, got {sorted(unknown)}")
    return local


def load_raw(lang: str, local: dict[str, str], repos: list[str], raw_dir: Path) -> tuple[list[dict], Path, dict]:
    """(raw rows, a JSONL file holding them, source info) for one language, from --input-jsonl or the Hub."""
    hf_split = HF_SPLIT[lang]
    if lang in local:
        raw_path = Path(local[lang])
        raw_rows = read_jsonl(raw_path)
        source = {"repo": f"local:{raw_path.name}", "split": hf_split, "loaded_as": "jsonl", "n_raw": len(raw_rows)}
        return raw_rows, raw_path, source
    raw_rows, source = load_from_hub(repos, hf_split)
    return raw_rows, write_jsonl(raw_dir / f"{hf_split}.jsonl", raw_rows), source


def split_mode_name(args) -> str:
    """How the splits are made, as recorded in the manifest."""
    if args.stratify:
        return "split_ids_stratified"
    if args.no_curatorkit:
        return "split_ids"
    return "curatorkit_output_split"


def build_language(
    args, lang: str, raw_rows: list[dict], raw_path: Path, source: dict, *, out_lang: Path, fractions: dict, seed: int
) -> tuple[dict[str, list[dict]], dict]:
    """Dedup and split one language by the path the flags choose; returns (splits, stats)."""
    if args.no_curatorkit:
        return build_plain(
            raw_rows,
            lang=lang,
            source=source,
            fractions=fractions,
            seed=seed,
            stratify=args.stratify,
            near_threshold=args.near_threshold,
        )
    return build_with_curatorkit(
        raw_path,
        out_lang / "_curatorkit",
        lang=lang,
        source=source,
        fractions=fractions,
        seed=seed,
        stratify=args.stratify,
        minhash_threshold=args.minhash_threshold,
    )


def write_splits(lang: str, splits: dict[str, list[dict]], out_lang: Path) -> None:
    """Record each row's split in its metadata and write one JSONL per split; raises if splits share an id."""
    all_ids = [row["id"] for rows in splits.values() for row in rows]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError(f"{lang}: an id appears in more than one split")
    for split, rows in splits.items():
        for row in rows:
            row["metadata"]["split"] = split
        write_jsonl(out_lang / f"{split}.jsonl", rows)


def update_manifest(path: Path, built: dict) -> None:
    """Write the manifest, keeping the entries of languages not rebuilt this time."""
    earlier = {}
    if path.exists():
        earlier = json.loads(path.read_text())["languages"]
    write_manifest(path, {"name": "harmful", "updated": timestamp(), "languages": {**earlier, **built}})


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    fractions = {name: float(value) for name, value in config["split_fractions"].items()}
    seed = config["split_seed"] if args.seed is None else args.seed
    data_dir = resolve_data_dir(args.data_dir)
    local = parse_local_inputs(args.input_jsonl)
    repos = hub_candidates(args.dataset, config["datasets"].get("harmful"), *FALLBACK_REPOS)
    split_mode = split_mode_name(args)

    built = {}
    for lang in args.langs:
        raw_rows, raw_path, source = load_raw(lang, local, repos, Path(args.raw_dir))
        out_lang = data_dir / "harmful" / lang
        splits, stats = build_language(
            args, lang, raw_rows, raw_path, source, out_lang=out_lang, fractions=fractions, seed=seed
        )
        write_splits(lang, splits, out_lang)
        built[lang] = {
            "built": timestamp(),
            "git": git_hash(),
            "curatorkit": not args.no_curatorkit,
            "split_mode": split_mode,
            "fractions": fractions,
            "seed": seed,
            "source": source,
            **stats,
            "counts": {split: len(rows) for split, rows in splits.items()},
            "harm_category_counts": {split: category_counts(rows) for split, rows in splits.items()},
        }
        counts = ", ".join(f"{split}={len(rows)}" for split, rows in splits.items())
        print(f"[harmful] {lang}: {counts} -> {out_lang}")

    manifest_path = data_dir / "harmful" / "manifest.json"
    update_manifest(manifest_path, built)
    print(f"[harmful] manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
