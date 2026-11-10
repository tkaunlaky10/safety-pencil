"""Data helpers shared by every script: the processed-data layout, ids, splits, contamination mixing and
overlap checks. Standard library only, so any script or test can import it.

Processed layout: UTF-8 JSONL under ``data/processed/`` (scripts take ``--data-dir``).

``harmful/{lang}/{factor,dev,test}.jsonl``, ``xstest/{lang}/{dev,test}.jsonl`` (lang: en, hi, hinglish)
    ``{"id", "prompt", "metadata"}``. ``metadata`` has ``lang``, ``split``, ``source`` (Hub id) and the
    source's own columns (aya_redteaming: ``harm_category`` as a list, ...; XSTest: ``xstest_type``, ...).
    Ids are ``stable_id(lang, prompt)``; a converted row keeps its source's id plus ``-hinglish`` or ``-hi``
    and records ``metadata.source_id`` / ``source_lang``. Hinglish rows are made per split from the en and
    hi rows of the same split, so a prompt and its Hinglish twin always share a split.
``{harmful,xstest}/{lang}/_curatorkit/``: CuratorKIT's own outputs (manifests, provenance, prompts_meta).
``{harmful,xstest,task_t1}/manifest.json``: our build record (sources, counts, seed, ...).
``task_t1/{train_4000,train_1000,heldout}.jsonl``
    ``{"id", "prompt", "answer"}``, Hindi rows of the Aya Dataset. train_1000 is a subset of train_4000;
    heldout (300 rows) is disjoint from both.
``contamination/beavertails_{n}.jsonl``
    ``{"id", "prompt", "answer", "metadata"}``: n BeaverTails harmful pairs for run D. Raw harmful
    completions, so the folder has its own ``.gitignore`` and is never committed (CLAUDE.md rule 7).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
import re
import tempfile
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed"

LANGS = ("en", "hi", "hinglish")
HARMFUL_SPLITS = ("factor", "dev", "test")
XSTEST_SPLITS = ("dev", "test")
PROMPT_SET_SPLITS = {"harmful": HARMFUL_SPLITS, "xstest": XSTEST_SPLITS}

NGRAM = 5  # character n-gram size for near-duplicate checks
MAX_EXAMPLES = 50  # matches listed per pair in an overlap report

# The script that builds each set, named in the error when a file is missing.
_BUILDER_SCRIPT = {
    "harmful": "scripts/01_build_harmful.py (and 01_make_hinglish.py for hinglish)",
    "xstest": "scripts/01_build_xstest.py",
    "task_t1": "scripts/01_build_task_t1.py",
    "contamination": "scripts/01_build_task_t1.py",
}
_TASK_T1_FILE_NAME = re.compile(r"train_\d+|heldout")
_CONTAMINATION_PREFIX = "beavertails_"

_WHITESPACE_RUN = re.compile(r"\s+")
_ID_HEX_DIGITS = 16
_FRACTION_SUM_TOLERANCE = 1e-6
_CEIL_SLACK = 1e-9  # keeps ceil(t * L) from rounding up when t * L is an integer plus float noise


# ══ Processed layout and JSONL I/O ═══════════════════════════════════════════


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    """All rows of a JSONL file (blank lines skipped); a bad line raises with its line number."""
    rows = []
    with open(path, encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_no}: {error}") from error
    return rows


def write_jsonl(path: str | os.PathLike, rows: Iterable[Mapping]) -> Path:
    """Write rows atomically (temp file + rename), creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        os.replace(tmp_path, path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    return path


def resolve_data_dir(data_dir: str | os.PathLike | None = None) -> Path:
    """The processed-data root: ``data_dir`` if given, else ``data/processed`` in the repo."""
    if data_dir is None:
        return DATA_DIR
    return Path(data_dir)


def split_path(data_dir: str | os.PathLike | None, name: str, split: str | int, lang: str | None = None) -> Path:
    """Path of one file in the processed layout (see the module docstring)."""
    root = resolve_data_dir(data_dir)
    if name in PROMPT_SET_SPLITS:
        return _prompt_set_path(root, name, split, lang)
    if name == "task_t1":
        return _task_t1_path(root, str(split), lang)
    if name == "contamination":
        return _contamination_path(root, str(split))
    raise ValueError(f"unknown set {name!r}; expected harmful, xstest, task_t1 or contamination")


def _prompt_set_path(root: Path, name: str, split: str, lang: str | None) -> Path:
    """``<root>/{harmful,xstest}/<lang>/<split>.jsonl``, after checking the language and split."""
    if lang not in LANGS:
        raise ValueError(f"{name} needs lang in {LANGS}, got {lang!r}")
    if split not in PROMPT_SET_SPLITS[name]:
        raise ValueError(f"{name} splits are {PROMPT_SET_SPLITS[name]}, got {split!r}")
    return root / name / lang / f"{split}.jsonl"


def _task_t1_path(root: Path, split: str, lang: str | None) -> Path:
    """``<root>/task_t1/{train_<n>,heldout}.jsonl``; task T1 is Hindi only."""
    if lang not in (None, "hi"):
        raise ValueError("task_t1 is Hindi only")
    if not _TASK_T1_FILE_NAME.fullmatch(split):
        raise ValueError(f"task_t1 files are train_<n> or heldout, got {split!r}")
    return root / "task_t1" / f"{split}.jsonl"


def _contamination_path(root: Path, split: str) -> Path:
    """``<root>/contamination/beavertails_<n>.jsonl``; ``split`` is n or the full file stem."""
    if not split.startswith(_CONTAMINATION_PREFIX):
        split = _CONTAMINATION_PREFIX + split
    return root / "contamination" / f"{split}.jsonl"


def load_split(data_dir: str | os.PathLike | None, name: str, split: str | int, lang: str | None = None) -> list[dict]:
    """Read one processed file, e.g. ``load_split(None, "harmful", "dev", "hi")``,
    ``load_split(None, "task_t1", "train_1000")`` or ``load_split(None, "contamination", 20)``."""
    path = split_path(data_dir, name, split, lang)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; build it with {_BUILDER_SCRIPT[name]}")
    return read_jsonl(path)


# ══ Build-script helpers ═════════════════════════════════════════════════════


def hub_candidates(*repos: str | None) -> list[str]:
    """The given Hub ids in order, without empty entries and repeats."""
    candidates = []
    for repo in repos:
        if repo and repo not in candidates:
            candidates.append(repo)
    return candidates


def write_manifest(path: str | os.PathLike, manifest: Mapping) -> Path:
    """Write a build record as indented UTF-8 JSON, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def timestamp() -> str:
    """Local time for build records, e.g. 2026-11-09T21:30:00+0530."""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# ══ Text normalisation and ids ═══════════════════════════════════════════════


def normalize_text(text: str) -> str:
    """NFC, whitespace runs collapsed to one space, stripped. Ids are computed from this."""
    if not text:
        return ""
    nfc_text = unicodedata.normalize("NFC", text)
    return _WHITESPACE_RUN.sub(" ", nfc_text).strip()


def normalize_for_match(text: str) -> str:
    """Aggressive normaliser for overlap checks: NFKC, casefold, punctuation and symbols (including the
    danda) removed, whitespace collapsed. Devanagari vowel signs are kept."""
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", text).casefold()
    kept_chars = []
    for char in folded:
        if _is_punctuation_or_symbol(char):
            kept_chars.append(" ")
        else:
            kept_chars.append(char)
    return _WHITESPACE_RUN.sub(" ", "".join(kept_chars)).strip()


def _is_punctuation_or_symbol(char: str) -> bool:
    """True for Unicode categories P* (punctuation, including the danda) and S* (symbols)."""
    major_category = unicodedata.category(char)[0]
    return major_category in ("P", "S")


def clean_prompt(text: str) -> str:
    """A one-line prompt gets ``normalize_text``; a multi-line one is only stripped, keeping its line breaks."""
    if "\n" in text:
        return text.strip()
    return normalize_text(text)


def stable_id(lang: str, text: str) -> str:
    """``"<lang>-<16 hex>"`` from the sha256 of the language tag and ``normalize_text(text)``."""
    key = f"{lang}\x1f{normalize_text(text)}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    return f"{lang}-{digest[:_ID_HEX_DIGITS]}"


# ══ Splits ═══════════════════════════════════════════════════════════════════
#
# split_ids gives each part an exact size (largest remainder), then fills the parts either by slicing one
# seeded shuffle of the sorted ids, or, with strata, by dealing the ids stratum by stratum so that every
# stratum is spread over the parts in about the global proportions.


def stratum_key(value: Any) -> str:
    """A hashable, order-free stratum label: a list becomes its sorted items joined by '|'."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set, frozenset)):
        return "|".join(sorted(str(item) for item in value))
    return str(value)


def split_ids(
    ids: Iterable[str], fractions: Mapping[str, float], seed: int, strata: Iterable[Any] | None = None
) -> dict[str, list[str]]:
    """Deterministic split of unique ids into named parts of exact (largest-remainder) sizes.

    The result depends only on the set of ids, the strata and the seed, not on input order. With
    ``strata`` (one label per id, e.g. the harm-category list) every stratum is split in about the global
    proportions (within ~2 rows). Each part is shuffled, so its first k ids are a random subset.
    Unlike CuratorKIT's ``output_split`` (a plain shuffle with floor-rounded sizes).
    """
    ids = list(ids)
    _check_unique(ids)
    part_names = list(fractions)
    part_fractions = [float(fractions[name]) for name in part_names]
    _check_fractions(part_names, part_fractions, fractions)
    part_sizes = _largest_remainder(len(ids), part_fractions)
    rng = random.Random(seed)
    if strata is None:
        parts = _slice_one_shuffle(ids, part_sizes, rng)
    else:
        parts = _deal_by_stratum(ids, list(strata), part_sizes, rng)
    return dict(zip(part_names, parts))


def _check_unique(ids: list[str]) -> None:
    """Raise if any id appears twice, naming up to five of the repeats."""
    if len(set(ids)) == len(ids):
        return
    repeated = [item for item, count in Counter(ids).items() if count > 1]
    raise ValueError(f"split_ids needs unique ids; duplicates include {repeated[:5]}")


def _check_fractions(names: list[str], values: list[float], fractions: Mapping[str, float]) -> None:
    """Raise unless there is at least one part and the fractions are non-negative and sum to 1."""
    has_negative = any(value < 0 for value in values)
    sum_error = abs(sum(values) - 1.0)
    if not names or has_negative or sum_error > _FRACTION_SUM_TOLERANCE:
        raise ValueError(f"fractions must be non-negative and sum to 1, got {dict(fractions)}")


def _largest_remainder(n: int, fractions: Sequence[float]) -> list[int]:
    """Integer part sizes that sum to n, each within 1 of ``n * fraction``.

    Every part gets floor(n · fraction); the rows left over go one each to the parts with the largest
    fractional remainders (ties to the earlier part).
    """
    exact_sizes = [n * fraction for fraction in fractions]
    sizes = [math.floor(size) for size in exact_sizes]
    remainders = [exact - floor for exact, floor in zip(exact_sizes, sizes)]
    by_remainder = sorted(range(len(fractions)), key=lambda j: (-remainders[j], j))
    n_left_over = n - sum(sizes)
    for j in by_remainder[:n_left_over]:
        sizes[j] += 1
    return sizes


def _slice_one_shuffle(ids: list[str], part_sizes: list[int], rng: random.Random) -> list[list[str]]:
    """Shuffle the sorted ids once and cut the result into consecutive parts of the given sizes."""
    order = sorted(ids)
    rng.shuffle(order)
    parts = []
    start = 0
    for size in part_sizes:
        parts.append(order[start : start + size])
        start += size
    return parts


def _deal_by_stratum(ids: list[str], strata: list[Any], counts: list[int], rng: random.Random) -> list[list[str]]:
    """Shuffle each stratum, line the strata up, and deal ids one by one to the part furthest behind its
    quota at that point, so each stratum lands in the parts in about the global proportions."""
    if len(strata) != len(ids):
        raise ValueError(f"strata has {len(strata)} labels for {len(ids)} ids")
    queue = _ids_grouped_by_stratum(ids, strata, rng)
    parts: list[list[str]] = [[] for _ in counts]
    for n_dealt, item in enumerate(queue, start=1):
        target = _part_furthest_behind(parts, counts, n_dealt, len(ids))
        parts[target].append(item)
    for part in parts:
        rng.shuffle(part)
    return parts


def _ids_grouped_by_stratum(ids: list[str], strata: list[Any], rng: random.Random) -> list[str]:
    """All ids, stratum after stratum (strata in label order), each stratum's ids in seeded random order."""
    label_of = {item: stratum_key(stratum) for item, stratum in zip(ids, strata)}
    groups: dict[str, list[str]] = defaultdict(list)
    for item in sorted(ids):
        groups[label_of[item]].append(item)
    queue: list[str] = []
    for label in sorted(groups):
        rng.shuffle(groups[label])
        queue.extend(groups[label])
    return queue


def _part_furthest_behind(parts: list[list[str]], quotas: list[int], n_dealt: int, n_total: int) -> int:
    """Index of the part, among those still below their quota, whose size lags most behind its pro-rata
    share quota · n_dealt / n_total; ties go to the earliest part."""
    candidates = []
    for j, quota in enumerate(quotas):
        if len(parts[j]) >= quota:
            continue
        shortfall = quota * n_dealt / n_total - len(parts[j])
        candidates.append((shortfall, -j))
    _, negative_index = max(candidates)
    return -negative_index


# ══ Training rows: chat messages and contamination mixing ════════════════════

_ANSWER_KEYS = ("answer", "response", "output")


def to_messages(row: Mapping, system: str | None = None) -> list[dict[str, str]]:
    """Chat messages for SFT from a {prompt, answer} row (also accepts response/output, and passes an
    existing ``messages`` list through). A prompt-only row gives one user turn."""
    messages = _row_messages(row)
    if system:
        without_system = [message for message in messages if message["role"] != "system"]
        messages = [{"role": "system", "content": system}] + without_system
    return messages


def _row_messages(row: Mapping) -> list[dict[str, str]]:
    """A copy of the row's ``messages``, or else a user turn from its prompt (or instruction) plus an
    assistant turn if it has an answer."""
    if row.get("messages"):
        return [dict(message) for message in row["messages"]]
    prompt = row.get("prompt", row.get("instruction"))
    if not prompt:
        raise ValueError(f"row {row.get('id')!r} has no prompt")
    messages = [{"role": "user", "content": prompt}]
    answer = _first_non_empty(row, _ANSWER_KEYS)
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def _first_non_empty(row: Mapping, keys: Sequence[str]) -> Any:
    """The value of the first key whose value is non-empty, or None."""
    for key in keys:
        if row.get(key):
            return row[key]
    return None


def contamination_count(n_train: int, frac: float) -> int:
    """Rows to add so they make up ``frac`` of the mixed set: round(frac * n / (1 - frac))."""
    if not 0.0 <= frac < 1.0:
        raise ValueError(f"contamination fraction must be in [0, 1), got {frac}")
    return int(math.floor(frac * n_train / (1.0 - frac) + 0.5))


def mix_contamination(
    train_rows: Sequence[Mapping], contam_rows: Sequence[Mapping], frac: float, seed: int
) -> list[dict]:
    """Insert the first ``contamination_count(len(train_rows), frac)`` contamination rows at seeded random
    positions. Train rows keep their relative order; inserted rows get ``is_contamination``."""
    n_added = contamination_count(len(train_rows), frac)
    if n_added > len(contam_rows):
        raise ValueError(f"need {n_added} contamination rows for frac={frac}, only {len(contam_rows)} given")
    if n_added == 0:
        return [dict(row) for row in train_rows]
    total = len(train_rows) + n_added
    insert_at = set(random.Random(seed).sample(range(total), n_added))
    contamination = iter(contam_rows[:n_added])
    train = iter(train_rows)
    mixed = []
    for position in range(total):
        if position in insert_at:
            mixed.append(dict(next(contamination), is_contamination=True))
        else:
            mixed.append(dict(next(train)))
    return mixed


# ══ Overlap checks ═══════════════════════════════════════════════════════════
#
# Texts are compared after normalize_for_match. Exact matches are found with a hash index. Near matches
# use the Jaccard similarity of character n-gram sets ("shingles") with prefix filtering, so only pairs
# that can reach the threshold are compared, and those comparisons are exact.


def char_ngrams(text: str, n: int = NGRAM) -> set[str]:
    """The set of character n-grams of ``text``; a text shorter than n is its own single n-gram."""
    if len(text) < n:
        if text:
            return {text}
        return set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard(a: set, b: set) -> float:
    """|a ∩ b| / |a ∪ b|, with two empty sets counted as identical (1.0)."""
    if not a and not b:
        return 1.0
    n_shared = len(a & b)
    return n_shared / (len(a) + len(b) - n_shared)


def _prefix_tokens(shingles: set[str], frequency: Counter, threshold: float) -> list[str]:
    """The rarest few n-grams of a set: two sets with Jaccard >= threshold must share one of them.

    A set of size L that reaches the threshold shares at least ⌈t·L⌉ of its n-grams, so its L − ⌈t·L⌉ + 1
    rarest n-grams (rarest first, ties by text) always include a shared one.
    """
    if not shingles:
        return []
    rarest_first = sorted(shingles, key=lambda token: (frequency[token], token))
    prefix_len = len(rarest_first) - math.ceil(threshold * len(rarest_first) - _CEIL_SLACK) + 1
    return rarest_first[: max(prefix_len, 1)]


def _near_pairs(a_sh: list[set[str]], b_sh: list[set[str]], threshold: float) -> list[tuple[int, int, float]]:
    """All (i, j, jaccard) with Jaccard(a_i, b_j) >= threshold. Prefix filtering: two sets that reach the
    threshold must share one of their rarest few n-grams, so only those candidates are compared (exact)."""
    frequency: Counter = Counter()
    for shingles in a_sh + b_sh:
        frequency.update(shingles)

    b_index_of_token: dict[str, list[int]] = defaultdict(list)
    for j, shingles in enumerate(b_sh):
        for token in _prefix_tokens(shingles, frequency, threshold):
            b_index_of_token[token].append(j)

    matches = []
    for i, shingles in enumerate(a_sh):
        candidates = set()
        for token in _prefix_tokens(shingles, frequency, threshold):
            candidates.update(b_index_of_token.get(token, ()))
        for j in sorted(candidates):
            similarity = jaccard(shingles, b_sh[j])
            if similarity >= threshold:
                matches.append((i, j, similarity))
    return matches


def find_near_duplicates(texts: Sequence[str], threshold: float = 0.85) -> list[tuple[int, int, float]]:
    """Pairs (i, j, jaccard), i < j, within one list whose character n-gram Jaccard on
    ``normalize_for_match`` text is >= threshold (exact duplicates included)."""
    shingles = [char_ngrams(normalize_for_match(text)) for text in texts]
    all_pairs = _near_pairs(shingles, shingles, threshold)
    return sorted((i, j, similarity) for i, j, similarity in all_pairs if i < j)


def _exact_matches(a: list[str], b: list[str]) -> list[tuple[int, int]]:
    """(i, j) with a[i] == b[j], ignoring empty strings."""
    b_positions: dict[str, list[int]] = defaultdict(list)
    for j, text in enumerate(b):
        if text:
            b_positions[text].append(j)
    matches = []
    for i, text in enumerate(a):
        if not text:
            continue
        for j in b_positions.get(text, ()):
            matches.append((i, j))
    return matches


def check_no_overlap(
    sets: Mapping[str, Sequence[str]],
    *,
    pairs: Iterable[tuple[str, str]] | None = None,
    near_threshold: float | None = None,
    ids: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Find texts shared between named sets, after ``normalize_for_match``.

    Counts exact matches and, if ``near_threshold`` is set, near matches (character n-gram Jaccard >=
    threshold, exact ones excluded) for each (a, b) in ``pairs`` (default: every unordered pair). Up to
    MAX_EXAMPLES matches per pair are listed by index, and by id when ``ids`` is given; the report never
    copies texts.
    """
    pairs = list(itertools.combinations(sets, 2)) if pairs is None else list(pairs)
    for a, b in pairs:
        if a not in sets or b not in sets:
            raise KeyError(f"unknown set in pair {(a, b)}")

    normalized = {name: [normalize_for_match(text) for text in texts] for name, texts in sets.items()}
    shingles = {}
    if near_threshold is not None:
        shingles = _shingles_of_paired_sets(normalized, pairs)
    pair_reports = [_pair_report(a, b, normalized, shingles, near_threshold, ids) for a, b in pairs]

    near_check_on = near_threshold is not None
    total_near = None
    ngram = None
    if near_check_on:
        total_near = sum(report["n_near"] for report in pair_reports)
        ngram = NGRAM
    return {
        "normalizer": "normalize_for_match",
        "near_threshold": near_threshold,
        "ngram": ngram,
        "sizes": {name: len(texts) for name, texts in sets.items()},
        "within_set_duplicates": {name: _count_repeats(texts) for name, texts in normalized.items()},
        "n_exact": sum(report["n_exact"] for report in pair_reports),
        "n_near": total_near,
        "pairs": pair_reports,
    }


def _shingles_of_paired_sets(normalized: dict[str, list[str]], pairs: list[tuple[str, str]]) -> dict:
    """Character n-gram sets of every text, for each set that appears in at least one pair."""
    shingles = {}
    for pair in pairs:
        for name in pair:
            if name not in shingles:
                shingles[name] = [char_ngrams(text) for text in normalized[name]]
    return shingles


def _pair_report(
    a: str,
    b: str,
    normalized: dict[str, list[str]],
    shingles: dict[str, list[set[str]]],
    near_threshold: float | None,
    ids: Mapping[str, Sequence[str]] | None,
) -> dict[str, Any]:
    """Exact (and, with a threshold, near) matches between sets a and b: counts plus the first examples.
    Near matches exclude exact ones and are listed most similar first."""
    exact = _exact_matches(normalized[a], normalized[b])
    report: dict[str, Any] = {
        "a": a,
        "b": b,
        "n_exact": len(exact),
        "exact": [{"a": _match_label(ids, a, i), "b": _match_label(ids, b, j)} for i, j in exact[:MAX_EXAMPLES]],
    }
    if near_threshold is None:
        return report

    near = []
    for i, j, similarity in _near_pairs(shingles[a], shingles[b], near_threshold):
        if normalized[a][i] != normalized[b][j]:
            near.append((i, j, similarity))
    near.sort(key=lambda match: -match[2])
    report["n_near"] = len(near)
    report["near"] = [
        {"a": _match_label(ids, a, i), "b": _match_label(ids, b, j), "jaccard": round(similarity, 4)}
        for i, j, similarity in near[:MAX_EXAMPLES]
    ]
    return report


def _match_label(ids: Mapping[str, Sequence[str]] | None, set_name: str, index: int) -> dict:
    """How a matched text is named in the report: its index, plus its id when ids are known for the set."""
    if ids is not None and set_name in ids:
        return {"index": index, "id": ids[set_name][index]}
    return {"index": index}


def _count_repeats(texts: list[str]) -> int:
    """How many non-empty texts repeat an earlier one."""
    non_empty = [text for text in texts if text]
    return len(non_empty) - len(set(non_empty))


# ══ Script and language heuristics (gates for LLM conversions, not for evaluation) ═

# Devanagari (U+0900–097F), Devanagari Extended (U+A8E0–A8FF) and Vedic Extensions (U+1CD0–1CFF).
_DEVANAGARI_CHAR = re.compile("[ऀ-ॿ꣠-ꣿ᳐-᳿]")
_LOWERCASE_WORD = re.compile(r"[a-z]+")
_CURLY_APOSTROPHE = "’"

_MIN_ASCII_LETTER_SHARE = 0.9  # looks_hinglish: letters that must be ASCII
_MIN_HINDI_WORD_SHARE = 0.1  # looks_hinglish: words that must be romanised Hindi function words

# Romanised Hindi function words and very common verbs that are not also common English words
# (so no "to", "the", "me", "main", "hi", "is", "us", "do", "par", "mat").
_HINDI_ROMAN_WORD_GROUPS = {
    "to be": "hai hain ho hoga hogi honge hota hoti hote tha thi raha rahi rahe",
    "postpositions": "ka ki ke ko se mein liye wala wali wale pehle baad saath bina",
    "I and we": "mai mujhe mujhse mujh mera meri mere hum hame hamein humein hamara hamari",
    "you": "tum tumhe tumko tumhara tumhari aap aapko aapka aapki aapke aapse apko apna apni apne",
    "this, that, they": "yeh ye woh wo voh vo unka unki unke uska uski uske iska iski iske inka usko isko unko inko",
    "relatives": "unhe inhe jinhe jo",
    "questions": "kya kaise kaisa kaisi kyun kyon kyu kab kahan kaha kaun kis kisi kise kisko kitna kitni kitne",
    "connectives": "aur lekin nahi nahin bhi toh sirf bahut zyada jyada jaise jab agar phir fir abhi",
    "quantities and places": "kuch koi sab ek wahan yahan",
    "karna, sakna, chahna": "karna karne karo karein kare karu karun karke kar kiya kiye "
    "sakta sakti sakte chahiye chahta chahti chahte",
    "other verbs": "batao bataiye bataye bata banaye banana banane banao "
    "dena lena diya liya gaya gayi jaana jana jata jati jate",
    "other words": "matlab accha achha acha theek thik haan bilkul wajah tarah tarike tarika",
}
HINDI_ROMAN_WORDS = frozenset(word for group in _HINDI_ROMAN_WORD_GROUPS.values() for word in group.split())

# Lower-case refusal phrases (curly apostrophes are folded to "'" before matching).
_REFUSAL_PATTERNS = (
    # English
    "i can't",
    "i cannot",
    "i can not",
    "i won't",
    "i will not",
    "i'm sorry",
    "i am sorry",
    "sorry, but",
    "i'm unable",
    "i am unable",
    "i apologi",
    "as an ai",
    "i must decline",
    "can't assist",
    "cannot assist",
    "can't help with",
    "cannot help with",
    "not able to help",
    # Hindi
    "माफ़",
    "माफ",
    "क्षमा",
    "मैं मदद नहीं",
    "मदद नहीं कर सक",
    # romanised Hindi
    "maaf kijiye",
    "maaf kijie",
    "maaf karein",
    "maaf kare",
    "maaf kariye",
    "maafi chahta",
    "maafi chahti",
    "kshama",
    "madad nahi kar sakta",
    "madad nahi kar sakti",
    "madad nahin kar sakta",
    "madad nahin kar sakti",
    "sahayata nahi kar",
)


def contains_devanagari(text: str) -> bool:
    """True if the text has at least one Devanagari character."""
    if not text:
        return False
    return _DEVANAGARI_CHAR.search(text) is not None


def devanagari_fraction(text: str) -> float:
    """Share of letters (vowel signs included) that are Devanagari."""
    if not text:
        return 0.0
    letters = [char for char in text if _is_letter_or_vowel_sign(char)]
    if not letters:
        return 0.0
    n_devanagari = sum(1 for char in letters if _DEVANAGARI_CHAR.match(char))
    return n_devanagari / len(letters)


def _is_letter_or_vowel_sign(char: str) -> bool:
    """Letters, plus combining marks (Mn, Mc), which is how Devanagari vowel signs are classed."""
    return char.isalpha() or unicodedata.category(char) in ("Mn", "Mc")


def looks_hinglish(text: str) -> bool:
    """No Devanagari, letters at least 90% ASCII, and at least 10% of the words (and at least one) are
    romanised Hindi function words."""
    if not text or contains_devanagari(text):
        return False
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    ascii_share = sum(char.isascii() for char in letters) / len(letters)
    if ascii_share < _MIN_ASCII_LETTER_SHARE:
        return False
    words = _LOWERCASE_WORD.findall(text.lower())
    n_hindi_words = sum(word in HINDI_ROMAN_WORDS for word in words)
    if n_hindi_words == 0:
        return False
    return n_hindi_words / len(words) >= _MIN_HINDI_WORD_SHARE


def looks_like_refusal(text: str, source: str | None = None) -> bool:
    """True if ``text`` contains a refusal phrase that ``source`` does not itself contain."""
    folded_text = _fold_for_refusal_match(text)
    folded_source = _fold_for_refusal_match(source)
    return any(phrase in folded_text and phrase not in folded_source for phrase in _REFUSAL_PATTERNS)


def _fold_for_refusal_match(text: str | None) -> str:
    """Casefolded text with curly apostrophes made straight; empty for None."""
    if not text:
        return ""
    return text.casefold().replace(_CURLY_APOSTROPHE, "'")
