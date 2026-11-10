"""Data helpers shared by every script. Standard library only, so any script or test can import it.

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

_BUILDERS = {
    "harmful": "scripts/01_build_harmful.py (and 01_make_hinglish.py for hinglish)",
    "xstest": "scripts/01_build_xstest.py",
    "task_t1": "scripts/01_build_task_t1.py",
    "contamination": "scripts/01_build_task_t1.py",
}
NGRAM = 5  # character n-gram size for near-duplicate checks
MAX_EXAMPLES = 50  # matches listed per pair in an overlap report


# ── text normalisation and ids ───────────────────────────────────────────────

_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """NFC, whitespace runs collapsed to one space, stripped. Ids are computed from this."""
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text or "")).strip()


def normalize_for_match(text: str) -> str:
    """Aggressive normaliser for overlap checks: NFKC, casefold, punctuation and symbols (including the
    danda) removed, whitespace collapsed. Devanagari vowel signs are kept."""
    t = unicodedata.normalize("NFKC", text or "").casefold()
    t = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in t)
    return _WS_RE.sub(" ", t).strip()


def clean_prompt(text: str) -> str:
    """A one-line prompt gets ``normalize_text``; a multi-line one is only stripped, keeping its line breaks."""
    return normalize_text(text) if "\n" not in text else text.strip()


def stable_id(lang: str, text: str) -> str:
    """``"<lang>-<16 hex>"`` from the sha256 of the language tag and ``normalize_text(text)``."""
    h = hashlib.sha256(f"{lang}\x1f{normalize_text(text)}".encode()).hexdigest()
    return f"{lang}-{h[:16]}"


# ── splits ───────────────────────────────────────────────────────────────────


def stratum_key(value: Any) -> str:
    """A hashable, order-free stratum label: a list becomes its sorted items joined by '|'."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set, frozenset)):
        return "|".join(sorted(str(v) for v in value))
    return str(value)


def _largest_remainder(n: int, fractions: Sequence[float]) -> list[int]:
    """Integer part sizes that sum to n, each within 1 of ``n * fraction``."""
    raw = [n * f for f in fractions]
    counts = [math.floor(x) for x in raw]
    order = sorted(range(len(raw)), key=lambda j: (-(raw[j] - counts[j]), j))
    for j in order[: n - sum(counts)]:
        counts[j] += 1
    return counts


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
    if len(set(ids)) != len(ids):
        dup = [k for k, c in Counter(ids).items() if c > 1][:5]
        raise ValueError(f"split_ids needs unique ids; duplicates include {dup}")
    names = list(fractions)
    fr = [float(fractions[k]) for k in names]
    if not names or any(f < 0 for f in fr) or abs(sum(fr) - 1.0) > 1e-6:
        raise ValueError(f"fractions must be non-negative and sum to 1, got {dict(fractions)}")
    counts = _largest_remainder(len(ids), fr)
    rng = random.Random(seed)
    if strata is None:
        order = sorted(ids)
        rng.shuffle(order)
        parts, start = {}, 0
        for k, c in zip(names, counts):
            parts[k] = order[start : start + c]
            start += c
        return parts
    return dict(zip(names, _deal_by_stratum(ids, list(strata), counts, rng)))


def _deal_by_stratum(ids: list[str], strata: list[Any], counts: list[int], rng: random.Random) -> list[list[str]]:
    """Shuffle each stratum, line the strata up, and deal ids one by one to the part furthest behind its
    quota at that point, so each stratum lands in the parts in about the global proportions."""
    if len(strata) != len(ids):
        raise ValueError(f"strata has {len(strata)} labels for {len(ids)} ids")
    label_of = {i: stratum_key(s) for i, s in zip(ids, strata)}
    groups: dict[str, list[str]] = defaultdict(list)
    for i in sorted(ids):
        groups[label_of[i]].append(i)
    queue: list[str] = []
    for key in sorted(groups):
        rng.shuffle(groups[key])
        queue.extend(groups[key])
    n = len(ids)
    parts: list[list[str]] = [[] for _ in counts]
    for pos, i in enumerate(queue, start=1):
        open_parts = [j for j in range(len(counts)) if len(parts[j]) < counts[j]]
        j = max(open_parts, key=lambda j: (counts[j] * pos / n - len(parts[j]), -j))
        parts[j].append(i)
    for p in parts:
        rng.shuffle(p)
    return parts


# ── JSONL and the processed layout ───────────────────────────────────────────


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    """All rows of a JSONL file (blank lines skipped); a bad line raises with its line number."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: {e}") from e
    return rows


def write_jsonl(path: str | os.PathLike, rows: Iterable[Mapping]) -> Path:
    """Write rows atomically (temp file + rename), creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def resolve_data_dir(data_dir: str | os.PathLike | None = None) -> Path:
    return Path(data_dir) if data_dir is not None else DATA_DIR


def split_path(data_dir: str | os.PathLike | None, name: str, split: str | int, lang: str | None = None) -> Path:
    """Path of one file in the processed layout (see the module docstring)."""
    root = resolve_data_dir(data_dir)
    if name in PROMPT_SET_SPLITS:
        if lang not in LANGS:
            raise ValueError(f"{name} needs lang in {LANGS}, got {lang!r}")
        if split not in PROMPT_SET_SPLITS[name]:
            raise ValueError(f"{name} splits are {PROMPT_SET_SPLITS[name]}, got {split!r}")
        return root / name / lang / f"{split}.jsonl"
    split = str(split)
    if name == "task_t1":
        if lang not in (None, "hi"):
            raise ValueError("task_t1 is Hindi only")
        if not re.fullmatch(r"train_\d+|heldout", split):
            raise ValueError(f"task_t1 files are train_<n> or heldout, got {split!r}")
        return root / "task_t1" / f"{split}.jsonl"
    if name == "contamination":
        stem = split if split.startswith("beavertails_") else f"beavertails_{split}"
        return root / "contamination" / f"{stem}.jsonl"
    raise ValueError(f"unknown set {name!r}; expected harmful, xstest, task_t1 or contamination")


def load_split(data_dir: str | os.PathLike | None, name: str, split: str | int, lang: str | None = None) -> list[dict]:
    """Read one processed file, e.g. ``load_split(None, "harmful", "dev", "hi")``,
    ``load_split(None, "task_t1", "train_1000")`` or ``load_split(None, "contamination", 20)``."""
    p = split_path(data_dir, name, split, lang)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; build it with {_BUILDERS[name]}")
    return read_jsonl(p)


# ── shared by the build scripts ──────────────────────────────────────────────


def hub_candidates(*repos: str | None) -> list[str]:
    """The given Hub ids in order, without empty entries and repeats."""
    return list(dict.fromkeys(r for r in repos if r))


def write_manifest(path: str | os.PathLike, manifest: Mapping) -> Path:
    """Write a build record as indented UTF-8 JSON, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def timestamp() -> str:
    """Local time for build records, e.g. 2026-11-09T21:30:00+0530."""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# ── SFT helpers ──────────────────────────────────────────────────────────────


def to_messages(row: Mapping, system: str | None = None) -> list[dict[str, str]]:
    """Chat messages for SFT from a {prompt, answer} row (also accepts response/output, and passes an
    existing ``messages`` list through). A prompt-only row gives one user turn."""
    if row.get("messages"):
        msgs = [dict(m) for m in row["messages"]]
    else:
        prompt = row.get("prompt", row.get("instruction"))
        if not prompt:
            raise ValueError(f"row {row.get('id')!r} has no prompt")
        msgs = [{"role": "user", "content": prompt}]
        answer = next((row[k] for k in ("answer", "response", "output") if row.get(k)), None)
        if answer is not None:
            msgs.append({"role": "assistant", "content": answer})
    if system:
        msgs = [{"role": "system", "content": system}] + [m for m in msgs if m["role"] != "system"]
    return msgs


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
    k = contamination_count(len(train_rows), frac)
    if k > len(contam_rows):
        raise ValueError(f"need {k} contamination rows for frac={frac}, only {len(contam_rows)} given")
    if k == 0:
        return [dict(r) for r in train_rows]
    added = iter([dict(r, is_contamination=True) for r in contam_rows[:k]])
    train = iter(train_rows)
    total = len(train_rows) + k
    at = set(random.Random(seed).sample(range(total), k))
    return [next(added) if i in at else dict(next(train)) for i in range(total)]


# ── overlap / contamination checks ───────────────────────────────────────────


def char_ngrams(text: str, n: int = NGRAM) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _near_pairs(a_sh: list[set[str]], b_sh: list[set[str]], threshold: float) -> list[tuple[int, int, float]]:
    """All (i, j, jaccard) with Jaccard(a_i, b_j) >= threshold. Prefix filtering: two sets that reach the
    threshold must share one of their rarest few n-grams, so only those candidates are compared (exact)."""
    freq: Counter = Counter()
    for s in a_sh + b_sh:
        freq.update(s)

    def prefix(s: set[str]) -> list[str]:
        if not s:
            return []
        toks = sorted(s, key=lambda t: (freq[t], t))
        p = len(toks) - math.ceil(threshold * len(toks) - 1e-9) + 1
        return toks[: max(p, 1)]

    index: dict[str, list[int]] = defaultdict(list)
    for j, s in enumerate(b_sh):
        for t in prefix(s):
            index[t].append(j)
    out = []
    for i, s in enumerate(a_sh):
        candidates = {j for t in prefix(s) for j in index.get(t, ())}
        for j in sorted(candidates):
            sim = jaccard(s, b_sh[j])
            if sim >= threshold:
                out.append((i, j, sim))
    return out


def find_near_duplicates(texts: Sequence[str], threshold: float = 0.85) -> list[tuple[int, int, float]]:
    """Pairs (i, j, jaccard), i < j, within one list whose character n-gram Jaccard on
    ``normalize_for_match`` text is >= threshold (exact duplicates included)."""
    sh = [char_ngrams(normalize_for_match(t)) for t in texts]
    return sorted((i, j, s) for i, j, s in _near_pairs(sh, sh, threshold) if i < j)


def _exact_matches(a: list[str], b: list[str]) -> list[tuple[int, int]]:
    """(i, j) with a[i] == b[j], ignoring empty strings."""
    where: dict[str, list[int]] = defaultdict(list)
    for j, t in enumerate(b):
        if t:
            where[t].append(j)
    return [(i, j) for i, t in enumerate(a) if t for j in where.get(t, ())]


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
    norm = {k: [normalize_for_match(t) for t in v] for k, v in sets.items()}
    in_pairs = dict.fromkeys(k for pair in pairs for k in pair)
    shingles = {k: [char_ngrams(t) for t in norm[k]] for k in in_pairs} if near_threshold is not None else {}

    def label(k: str, i: int) -> dict:
        return {"index": i, "id": ids[k][i]} if ids and k in ids else {"index": i}

    report_pairs = []
    for a, b in pairs:
        exact = _exact_matches(norm[a], norm[b])
        entry: dict[str, Any] = {
            "a": a,
            "b": b,
            "n_exact": len(exact),
            "exact": [{"a": label(a, i), "b": label(b, j)} for i, j in exact[:MAX_EXAMPLES]],
        }
        if near_threshold is not None:
            near = [
                (i, j, s)
                for i, j, s in _near_pairs(shingles[a], shingles[b], near_threshold)
                if norm[a][i] != norm[b][j]
            ]
            near.sort(key=lambda x: -x[2])
            entry["n_near"] = len(near)
            entry["near"] = [
                {"a": label(a, i), "b": label(b, j), "jaccard": round(s, 4)} for i, j, s in near[:MAX_EXAMPLES]
            ]
        report_pairs.append(entry)

    nonempty = {k: [t for t in v if t] for k, v in norm.items()}
    return {
        "normalizer": "normalize_for_match",
        "near_threshold": near_threshold,
        "ngram": NGRAM if near_threshold is not None else None,
        "sizes": {k: len(v) for k, v in sets.items()},
        "within_set_duplicates": {k: len(v) - len(set(v)) for k, v in nonempty.items()},
        "n_exact": sum(p["n_exact"] for p in report_pairs),
        "n_near": sum(p["n_near"] for p in report_pairs) if near_threshold is not None else None,
        "pairs": report_pairs,
    }


# ── script / language heuristics (gates for LLM conversions, not for evaluation) ─

_DEVANAGARI_RE = re.compile("[ऀ-ॿ꣠-ꣿ᳐-᳿]")  # Devanagari, its Extended, Vedic

# Romanised Hindi function words and very common verbs that are not also common English words
# (so no "to", "the", "me", "main", "hi", "is", "us", "do", "par", "mat").
HINDI_ROMAN_WORDS = frozenset(
    [
        "hai",
        "hain",
        "ho",
        "hoga",
        "hogi",
        "honge",
        "hota",
        "hoti",
        "hote",
        "tha",
        "thi",
        "raha",
        "rahi",
        "rahe",
        "ka",
        "ki",
        "ke",
        "ko",
        "se",
        "mein",
        "mai",
        "mujhe",
        "mujhse",
        "mujh",
        "mera",
        "meri",
        "mere",
        "hum",
        "hame",
        "hamein",
        "humein",
        "hamara",
        "hamari",
        "tum",
        "tumhe",
        "tumko",
        "tumhara",
        "tumhari",
        "aap",
        "aapko",
        "aapka",
        "aapki",
        "aapke",
        "aapse",
        "apko",
        "apna",
        "apni",
        "apne",
        "yeh",
        "ye",
        "woh",
        "wo",
        "voh",
        "vo",
        "kya",
        "kaise",
        "kaisa",
        "kaisi",
        "kyun",
        "kyon",
        "kyu",
        "kab",
        "kahan",
        "kaha",
        "kaun",
        "kis",
        "kisi",
        "kise",
        "kisko",
        "kuch",
        "koi",
        "sab",
        "aur",
        "lekin",
        "nahi",
        "nahin",
        "bhi",
        "toh",
        "sirf",
        "bahut",
        "zyada",
        "jyada",
        "liye",
        "wala",
        "wali",
        "wale",
        "karna",
        "karne",
        "karo",
        "karein",
        "kare",
        "karu",
        "karun",
        "karke",
        "kar",
        "kiya",
        "kiye",
        "sakta",
        "sakti",
        "sakte",
        "chahiye",
        "chahta",
        "chahti",
        "chahte",
        "batao",
        "bataiye",
        "bataye",
        "bata",
        "banaye",
        "banana",
        "banane",
        "banao",
        "jaise",
        "jab",
        "agar",
        "phir",
        "fir",
        "abhi",
        "unka",
        "unki",
        "unke",
        "uska",
        "uski",
        "uske",
        "iska",
        "iski",
        "iske",
        "inka",
        "usko",
        "isko",
        "unko",
        "inko",
        "unhe",
        "inhe",
        "jinhe",
        "jo",
        "ek",
        "wahan",
        "yahan",
        "dena",
        "lena",
        "diya",
        "liya",
        "gaya",
        "gayi",
        "jaana",
        "jana",
        "jata",
        "jati",
        "jate",
        "kitna",
        "kitni",
        "kitne",
        "matlab",
        "accha",
        "achha",
        "acha",
        "theek",
        "thik",
        "haan",
        "bilkul",
        "pehle",
        "baad",
        "saath",
        "bina",
        "wajah",
        "tarah",
        "tarike",
        "tarika",
    ]
)

_REFUSAL_PATTERNS = (
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
    "माफ़",
    "माफ",
    "क्षमा",
    "मैं मदद नहीं",
    "मदद नहीं कर सक",
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
    return bool(_DEVANAGARI_RE.search(text or ""))


def devanagari_fraction(text: str) -> float:
    """Share of letters (vowel signs included) that are Devanagari."""
    letters = [c for c in text or "" if c.isalpha() or unicodedata.category(c) in ("Mn", "Mc")]
    if not letters:
        return 0.0
    return sum(1 for c in letters if _DEVANAGARI_RE.match(c)) / len(letters)


def looks_hinglish(text: str) -> bool:
    """No Devanagari, letters at least 90% ASCII, and at least 10% of the words (and at least one) are
    romanised Hindi function words."""
    if not text or contains_devanagari(text):
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters or sum(c.isascii() for c in letters) / len(letters) < 0.9:
        return False
    words = re.findall(r"[a-z]+", text.lower())
    hits = sum(w in HINDI_ROMAN_WORDS for w in words)
    return hits >= 1 and hits / max(len(words), 1) >= 0.1


def looks_like_refusal(text: str, source: str | None = None) -> bool:
    """True if ``text`` contains a refusal phrase that ``source`` does not itself contain."""
    t = (text or "").casefold().replace("’", "'")
    s = (source or "").casefold().replace("’", "'")
    return any(p in t and p not in s for p in _REFUSAL_PATTERNS)
