"""Data helpers shared by every script. Standard library only, so any script or test can import it.

Processed layout
================
Everything lives under ``data/processed/`` (``DATA_DIR``; scripts take ``--data-dir``). All files are
UTF-8 JSONL, one object per line.

``harmful/{lang}/{split}.jsonl``  lang in en, hi, hinglish; split in factor, dev, test
``xstest/{lang}/{split}.jsonl``   lang in en, hi, hinglish; split in dev, test
    ``{"id": str, "prompt": str, "metadata": {...}}``

    - ``metadata`` always has ``lang`` (en/hi/hinglish), ``split`` and ``source`` (Hub id). Harmful
      rows keep aya_redteaming's columns (``harm_category`` as a list, ``global_or_local``,
      ``literal_translation``, ...); XSTest rows keep ``xstest_type``, ``focus``, ``note``.
    - ids: ``stable_id(lang, prompt)`` = ``"<lang>-<16 hex>"``, computed from the source text.
      Converted rows keep their source's id with a suffix: ``"<source_id>-hinglish"``,
      ``"<source_id>-hi"``, and carry ``metadata.source_id`` / ``metadata.source_lang``.
    - Hinglish rows are made per split from the en and hi files of the same split, never by
      re-splitting, so a prompt and its Hinglish twin always share a split.

``harmful/{lang}/_curatorkit/``, ``xstest/{lang}/_curatorkit/``
    CuratorKIT's own outputs (manifest.json, lexsi_provenance.json, checksums, per-split
    prompts_meta.jsonl), kept next to the normalised files. No raw LLM outputs are kept here.
``harmful/manifest.json``, ``xstest/manifest.json``, ``task_t1/manifest.json``
    Our build record: sources, counts, fractions, seed, per-split category counts.

``task_t1/train_4000.jsonl``, ``task_t1/train_1000.jsonl``, ``task_t1/heldout.jsonl``
    ``{"id": str, "prompt": str, "answer": str}`` (Hindi rows of the Aya Dataset).
    train_1000 is a subset of train_4000; heldout (300 rows) is disjoint from both.
    ``to_messages(row)`` turns a row into chat messages for SFT.
``contamination/beavertails_{n}.jsonl``
    ``{"id", "prompt", "answer", "metadata"}``: n BeaverTails harmful pairs from SafeTune's
    ``harden_contamination_pairs(n)[0]``, for run D. Raw harmful completions: the folder carries
    its own ``.gitignore`` and must never be committed or pushed (CLAUDE.md rule 7).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed"

LANGS = ("en", "hi", "hinglish")
HARMFUL_SPLITS = ("factor", "dev", "test")
XSTEST_SPLITS = ("dev", "test")
T1_SPLITS = ("train_4000", "train_1000", "heldout")
PROMPT_SETS = ("harmful", "xstest")

_BUILDERS = {
    "harmful": "scripts/01_build_harmful.py (and 01_make_hinglish.py for hinglish)",
    "xstest": "scripts/01_build_xstest.py",
    "task_t1": "scripts/01_build_task_t1.py",
    "contamination": "scripts/01_build_task_t1.py",
}


# ── text normalisation and ids ───────────────────────────────────────────────

_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """NFC, all whitespace runs collapsed to one space, stripped. What ids are computed from."""
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text or "")).strip()


def normalize_for_match(text: str) -> str:
    """Aggressive normaliser for overlap checks: NFKC, casefold, punctuation and symbols removed
    (including the danda), whitespace collapsed. Devanagari vowel signs are kept."""
    t = unicodedata.normalize("NFKC", text or "").casefold()
    t = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in t)
    return _WS_RE.sub(" ", t).strip()


def stable_id(lang: str, text: str, length: int = 16) -> str:
    """Deterministic id ``"<lang>-<hex>"``: sha256 of the language tag and the NFC,
    whitespace-collapsed text. Insensitive to whitespace differences only."""
    h = hashlib.sha256(f"{lang}\x1f{normalize_text(text)}".encode("utf-8")).hexdigest()
    return f"{lang}-{h[:length]}"


# ── splits ───────────────────────────────────────────────────────────────────

def stratum_key(value: Any) -> str:
    """A hashable, order-free stratum label: lists become their sorted items joined by '|'."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set, frozenset)):
        return "|".join(sorted(str(v) for v in value))
    return str(value)


def _largest_remainder(n: int, fractions: Sequence[float]) -> list[int]:
    raw = [n * f for f in fractions]
    counts = [math.floor(x) for x in raw]
    order = sorted(range(len(raw)), key=lambda j: (-(raw[j] - counts[j]), j))
    for j in order[: n - sum(counts)]:
        counts[j] += 1
    return counts


def split_ids(ids: Iterable[str], fractions: Mapping[str, float], seed: int,
              strata: Iterable[Any] | None = None) -> dict[str, list[str]]:
    """Deterministic split of unique ids into named parts.

    - Depends only on the *set* of ids (they are sorted first), the strata labels and the seed,
      not on input order.
    - Part sizes use largest remainders, so each is within 1 of ``fraction * n`` and they sum to n.
    - With ``strata`` (one label per id, e.g. the harm-category list), ids are grouped by stratum,
      shuffled within each, and dealt to parts by sequential quota apportionment, so every
      stratum is split in about the global proportions (within ~2 rows) while part sizes stay exact.
    - Each part comes back in shuffled order, so its first k ids are a random subset.

    Note: this is not CuratorKIT's ``output_split`` (a shuffle with floor-rounded part sizes).
    """
    ids = list(ids)
    if len(set(ids)) != len(ids):
        dup = [k for k, c in Counter(ids).items() if c > 1][:5]
        raise ValueError(f"split_ids needs unique ids; duplicates include {dup}")
    names = list(fractions)
    fr = [float(fractions[k]) for k in names]
    if not names or any(f < 0 for f in fr) or abs(sum(fr) - 1.0) > 1e-6:
        raise ValueError(f"fractions must be non-negative and sum to 1, got {dict(fractions)}")
    n = len(ids)
    counts = _largest_remainder(n, fr)
    rng = random.Random(seed)

    if strata is None:
        order = sorted(ids)
        rng.shuffle(order)
        out, start = {}, 0
        for k, c in zip(names, counts):
            out[k] = order[start:start + c]
            start += c
        return out

    labels = [stratum_key(s) for s in strata]
    if len(labels) != n:
        raise ValueError(f"strata has {len(labels)} labels for {n} ids")
    label_of = dict(zip(ids, labels))
    groups: dict[str, list[str]] = defaultdict(list)
    for i in sorted(ids):
        groups[label_of[i]].append(i)
    seq: list[str] = []
    for key in sorted(groups):
        g = groups[key]
        rng.shuffle(g)
        seq.extend(g)
    assigned = [0] * len(names)
    parts: list[list[str]] = [[] for _ in names]
    for pos, i in enumerate(seq, start=1):
        j = max((j for j in range(len(names)) if assigned[j] < counts[j]),
                key=lambda j: (counts[j] * pos / n - assigned[j], -j))
        assigned[j] += 1
        parts[j].append(i)
    for p in parts:
        rng.shuffle(p)
    return dict(zip(names, parts))


# ── JSONL and the processed layout ───────────────────────────────────────────

def read_jsonl(path: str | os.PathLike) -> list[dict]:
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


def split_path(data_dir: str | os.PathLike | None, name: str, split: str | int,
               lang: str | None = None) -> Path:
    """Path of one file in the processed layout (see the module docstring)."""
    root = resolve_data_dir(data_dir)
    if name in PROMPT_SETS:
        if lang not in LANGS:
            raise ValueError(f"{name} needs lang in {LANGS}, got {lang!r}")
        valid = HARMFUL_SPLITS if name == "harmful" else XSTEST_SPLITS
        if split not in valid:
            raise ValueError(f"{name} splits are {valid}, got {split!r}")
        return root / name / lang / f"{split}.jsonl"
    if name == "task_t1":
        if lang not in (None, "hi"):
            raise ValueError("task_t1 is Hindi only")
        split = str(split)
        if not re.fullmatch(r"train_\d+|heldout", split):
            raise ValueError(f"task_t1 files are train_<n> or heldout, got {split!r}")
        return root / "task_t1" / f"{split}.jsonl"
    if name == "contamination":
        split = str(split)
        stem = split if split.startswith("beavertails_") else f"beavertails_{split}"
        return root / "contamination" / f"{stem}.jsonl"
    raise ValueError(f"unknown set {name!r}; expected harmful, xstest, task_t1 or contamination")


def load_split(data_dir: str | os.PathLike | None, name: str, split: str | int,
               lang: str | None = None) -> list[dict]:
    """Read one processed file, e.g. ``load_split(None, "harmful", "dev", "hi")``,
    ``load_split(None, "task_t1", "train_1000")`` or ``load_split(None, "contamination", 20)``."""
    p = split_path(data_dir, name, split, lang)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; build it with {_BUILDERS[name]}")
    return read_jsonl(p)


# ── SFT helpers ──────────────────────────────────────────────────────────────

def to_messages(row: Mapping, system: str | None = None) -> list[dict[str, str]]:
    """Chat messages for SFT from a {prompt, answer} row (also accepts response/output, and
    passes an existing ``messages`` list through). Prompt-only rows give one user turn."""
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


def mix_contamination(train_rows: Sequence[Mapping], contam_rows: Sequence[Mapping], frac: float,
                      seed: int) -> list[dict]:
    """Add the first ``contamination_count(len(train_rows), frac)`` contamination rows at seeded
    random positions. Train rows keep their relative order; added rows get ``is_contamination``."""
    k = contamination_count(len(train_rows), frac)
    if k > len(contam_rows):
        raise ValueError(f"need {k} contamination rows for frac={frac}, only {len(contam_rows)} given")
    if k == 0:
        return [dict(r) for r in train_rows]
    added = [dict(r, is_contamination=True) for r in contam_rows[:k]]
    total = len(train_rows) + k
    at = set(random.Random(seed).sample(range(total), k))
    tr, ad = iter(train_rows), iter(added)
    return [next(ad) if i in at else dict(next(tr)) for i in range(total)]


# ── overlap / contamination checks ───────────────────────────────────────────

def char_ngrams(text: str, n: int = 5) -> set[str]:
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _near_pairs(a_sh: list[set[str]], b_sh: list[set[str]], threshold: float,
                skip: set[tuple[int, int]]) -> list[tuple[int, int, float]]:
    """All (i, j) with Jaccard(a_i, b_j) >= threshold, by prefix filtering (exact, no misses)."""
    freq: Counter = Counter()
    for s in a_sh + b_sh:
        freq.update(s)

    def prefix(s: set[str]) -> list[str]:
        if not s:
            return []
        toks = sorted(s, key=lambda t: (freq[t], t))
        p = len(toks) - math.ceil(threshold * len(toks) - 1e-9) + 1
        return toks[:max(p, 1)]

    index: dict[str, list[int]] = defaultdict(list)
    for j, s in enumerate(b_sh):
        for t in prefix(s):
            index[t].append(j)
    out = []
    for i, s in enumerate(a_sh):
        cands = set()
        for t in prefix(s):
            cands.update(index.get(t, ()))
        for j in sorted(cands):
            if (i, j) in skip:
                continue
            sim = jaccard(s, b_sh[j])
            if sim >= threshold:
                out.append((i, j, sim))
    return out


def find_near_duplicates(texts: Sequence[str], threshold: float = 0.85, ngram: int = 5,
                         normalizer: Callable[[str], str] = normalize_for_match) -> list[tuple[int, int, float]]:
    """Pairs (i, j, jaccard) with i < j inside one list whose character n-gram Jaccard on the
    normalised texts is >= threshold (exact duplicates included)."""
    sh = [char_ngrams(normalizer(t or ""), ngram) for t in texts]
    return sorted((i, j, s) for i, j, s in _near_pairs(sh, sh, threshold, set()) if i < j)


def check_no_overlap(sets: Mapping[str, Sequence[str]],
                     normalizer: Callable[[str], str] = normalize_for_match, *,
                     pairs: Iterable[tuple[str, str]] | None = None,
                     near_threshold: float | None = None, ngram: int = 5,
                     ids: Mapping[str, Sequence[str]] | None = None,
                     max_examples: int = 50) -> dict[str, Any]:
    """Find texts shared between named sets.

    - exact: equal after ``normalizer`` (default ``normalize_for_match``).
    - near (only if ``near_threshold`` is set): character ``ngram``-gram Jaccard >= threshold on
      the normalised texts, excluding exact matches. Pure Python with prefix filtering; fine for
      tens of thousands of rows.

    ``pairs`` restricts which (a, b) set pairs are compared (default: every unordered pair).
    ``ids`` (same keys and lengths as ``sets``) labels matches with ids instead of only indices.
    Returns counts per pair and overall, plus up to ``max_examples`` matches per pair; texts are
    never copied into the report.
    """
    names = list(sets)
    if pairs is None:
        pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    pairs = list(pairs)
    for a, b in pairs:
        if a not in sets or b not in sets:
            raise KeyError(f"unknown set in pair {(a, b)}")
    norm = {k: [normalizer(t or "") for t in v] for k, v in sets.items()}
    shingles: dict[str, list[set[str]]] = {}

    def label(k, i):
        return {"index": i, "id": ids[k][i]} if ids and k in ids else {"index": i}

    report_pairs = []
    tot_exact = tot_near = 0
    for a, b in pairs:
        where: dict[str, list[int]] = defaultdict(list)
        for j, t in enumerate(norm[b]):
            if t:
                where[t].append(j)
        exact = [(i, j) for i, t in enumerate(norm[a]) if t for j in where.get(t, ())]
        entry: dict[str, Any] = {
            "a": a, "b": b, "n_exact": len(exact),
            "exact": [{"a": label(a, i), "b": label(b, j)} for i, j in exact[:max_examples]],
        }
        tot_exact += len(exact)
        if near_threshold is not None:
            for k in (a, b):
                if k not in shingles:
                    shingles[k] = [char_ngrams(t, ngram) for t in norm[k]]
            near = _near_pairs(shingles[a], shingles[b], near_threshold, set(exact))
            near = [(i, j, s) for i, j, s in near if norm[a][i] != norm[b][j]]
            near.sort(key=lambda x: -x[2])
            entry["n_near"] = len(near)
            entry["near"] = [{"a": label(a, i), "b": label(b, j), "jaccard": round(s, 4)}
                             for i, j, s in near[:max_examples]]
            tot_near += len(near)
        report_pairs.append(entry)

    within = {}
    for k, v in norm.items():
        nonempty = [t for t in v if t]
        within[k] = len(nonempty) - len(set(nonempty))
    return {
        "normalizer": getattr(normalizer, "__name__", str(normalizer)),
        "near_threshold": near_threshold,
        "ngram": ngram if near_threshold is not None else None,
        "sizes": {k: len(v) for k, v in sets.items()},
        "within_set_duplicates": within,
        "n_exact": tot_exact,
        "n_near": tot_near if near_threshold is not None else None,
        "pairs": report_pairs,
    }


# ── script / language heuristics (for gates, not for evaluation) ─────────────

_DEVANAGARI_RE = re.compile("[ऀ-ॿ꣠-ꣿ᳐-᳿]")

# Romanised Hindi function words and very common verbs that are not also common English words
# (so no "to", "the", "me", "main", "hi", "is", "us", "do", "par", "mat").
HINDI_ROMAN_WORDS = frozenset("""
hai hain ho hoga hogi honge hota hoti hote tha thi raha rahi rahe
ka ki ke ko se mein mai mujhe mujhse mujh mera meri mere hum hame hamein humein hamara hamari
tum tumhe tumko tumhara tumhari aap aapko aapka aapki aapke aapse apko apna apni apne
yeh ye woh wo voh vo kya kaise kaisa kaisi kyun kyon kyu kab kahan kaha kaun kis kisi kise kisko
kuch koi sab aur lekin nahi nahin bhi toh sirf bahut zyada jyada liye wala wali wale
karna karne karo karein kare karu karun karke kar kiya kiye sakta sakti sakte chahiye chahta
chahti chahte batao bataiye bataye bata banaye banana banane banao jaise jab agar phir fir abhi
unka unki unke uska uski uske iska iski iske inka usko isko unko inko unhe inhe jinhe jo ek
wahan yahan dena lena diya liya gaya gayi jaana jana jata jati jate kitna kitni kitne matlab
accha achha acha theek thik haan bilkul pehle baad saath bina wajah tarah tarike tarika
""".split())

_REFUSAL_PATTERNS = (
    "i can't", "i cannot", "i can not", "i won't", "i will not", "i'm sorry", "i am sorry",
    "sorry, but", "i'm unable", "i am unable", "i apologi", "as an ai", "i must decline",
    "can't assist", "cannot assist", "can't help with", "cannot help with", "not able to help",
    "माफ़", "माफ", "क्षमा", "मैं मदद नहीं", "मदद नहीं कर सक",
    "maaf kijiye", "maaf kijie", "maaf karein", "maaf kare", "maaf kariye", "maafi chahta",
    "maafi chahti", "kshama", "madad nahi kar sakta", "madad nahi kar sakti",
    "madad nahin kar sakta", "madad nahin kar sakti", "sahayata nahi kar",
)


def contains_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI_RE.search(text or ""))


def devanagari_fraction(text: str) -> float:
    """Share of letters (including vowel signs) that are Devanagari."""
    letters = [c for c in text or "" if c.isalpha() or unicodedata.category(c) in ("Mn", "Mc")]
    if not letters:
        return 0.0
    return sum(1 for c in letters if _DEVANAGARI_RE.match(c)) / len(letters)


def looks_hinglish(text: str, min_hits: int = 1, min_ratio: float = 0.1) -> bool:
    """Heuristic: no Devanagari, letters mostly ASCII, and at least ``min_hits`` (and ``min_ratio``
    of the words) are romanised Hindi function words."""
    if not text or contains_devanagari(text):
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters or sum(c.isascii() for c in letters) / len(letters) < 0.9:
        return False
    toks = re.findall(r"[a-z]+", text.lower())
    hits = sum(t in HINDI_ROMAN_WORDS for t in toks)
    return hits >= min_hits and hits / max(len(toks), 1) >= min_ratio


def looks_like_refusal(text: str, source: str | None = None) -> bool:
    """True if ``text`` contains a refusal phrase that the source text does not itself contain."""
    t = (text or "").casefold().replace("’", "'")
    s = (source or "").casefold().replace("’", "'")
    return any(p in t and p not in s for p in _REFUSAL_PATTERNS)
