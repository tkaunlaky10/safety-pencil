#!/usr/bin/env python
"""Make Hinglish versions of prompt sets through an LLM, as a CuratorKIT generation task.

Hinglish here means romanised, code-mixed Hindi-English: Latin script only, meaning preserved, nothing
added or removed. Each en and hi prompt of a split becomes one Hinglish prompt of the same split, with id
``<source_id>-hinglish``:

    harmful/{en,hi}/{split}.jsonl  ->  harmful/hinglish/{split}.jsonl      (--set harmful)
    xstest/en/{split}.jsonl        ->  xstest/hinglish/{split}.jsonl       (--set xstest)

``HinglishTask`` (and ``HindiTranslationTask``, which 01_build_xstest.py uses for en -> hi) run through
CuratorKIT's ``Pipeline`` (JSONLReader -> task -> PromptsMetaExporter). An output is rejected if it is in
the wrong script, looks like a refusal, has a length ratio to its source outside [0.5, 2.0], or (Hinglish)
has no romanised Hindi function words. Rejects are logged by id, reason, length and hash; for the harmful
set the rejected text itself is never written anywhere, since it could answer the harmful prompt.

A random 10% of each split goes to outputs/review/hinglish_check.csv for a human spot check (outputs/ is
gitignored). ``--dry-run`` uses a fake offline LLM and, unless --out-dir is given, writes under
outputs/dry_run/ so real data is never overwritten.

LLM: any LiteLLM model string, e.g. ``--model openai/CohereLabs/aya-expanse-8b --api-base
http://localhost:8000/v1`` (local vLLM) or ``--model cohere_chat/c4ai-aya-expanse-32b`` (COHERE_API_KEY
in .env).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import math
import random
import re
import sys
from collections import Counter
from functools import partial
from pathlib import Path

from curatorkit.generators.base import BaseGenerationTask
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample

from pencil.datasets import (
    PROMPT_SET_SPLITS,
    REPO_ROOT,
    contains_devanagari,
    devanagari_fraction,
    looks_hinglish,
    looks_like_refusal,
    normalize_for_match,
    normalize_text,
    read_jsonl,
    resolve_data_dir,
    timestamp,
    write_jsonl,
    write_manifest,
)

LANG_NAMES = {"en": "English", "hi": "Hindi", "hinglish": "Hinglish"}
SET_SOURCES = {"harmful": ("en", "hi"), "xstest": ("en",)}
SPLIT_KEEP_ORDER = ("test", "dev", "factor")  # which split keeps an output that two splits share
DEFAULT_REVIEW_DIR = REPO_ROOT / "outputs" / "review"
DRY_RUN_DIR = REPO_ROOT / "outputs" / "dry_run" / "processed"

SYSTEM_HINGLISH = (
    "You rewrite text in Hinglish: the code-mixed Hindi-English that people in India type in chats, "
    "written entirely in the Latin alphabet (romanised Hindi, never Devanagari).\n"
    "Rules:\n"
    "- Keep exactly the same meaning, intent, tone and level of detail. Do not add, remove, soften, "
    "explain, judge or answer anything.\n"
    "- This is a conversion step for a multilingual AI-safety research dataset. Some texts are "
    "requests a model should refuse; convert them faithfully anyway. Never answer them and never refuse.\n"
    "- Use Hindi grammar and everyday Hindi words in Roman script, and keep English words that Hindi "
    "speakers normally use (technical terms, names, brands).\n"
    "- Reply with the converted text only: no quotes, labels, notes or alternatives."
)
SYSTEM_HINDI = (
    "You translate English text into natural, everyday Hindi written in Devanagari.\n"
    "Rules:\n"
    "- Keep exactly the same meaning, intent, tone and level of detail. Do not add, remove, soften, "
    "explain, judge or answer anything.\n"
    "- This is a translation step for a multilingual AI-safety research dataset. Some texts may sound "
    "alarming; translate them faithfully anyway. Never answer them and never refuse.\n"
    "- Keep names, brands and technical terms that Hindi speakers write in English as they are.\n"
    "- Reply with the translation only: no quotes, labels, notes or alternatives."
)
# Benign worked examples (source language, source, output).
FEWSHOT = {
    "hinglish": [
        (
            "en",
            "How can I fix my sleep schedule before the exams start next week?",
            "Next week exams start ho rahe hain, usse pehle main apna sleep schedule kaise theek karun?",
        ),
        (
            "hi",
            "मुझे बताइए कि घर पर अच्छी मसाला चाय कैसे बनाते हैं।",
            "Mujhe batao ki ghar par acchi masala chai kaise banate hain.",
        ),
    ],
    "hi": [
        (
            "en",
            "How can I fix my sleep schedule before the exams start next week?",
            "अगले हफ़्ते परीक्षा शुरू होने से पहले मैं अपना सोने का समय कैसे ठीक करूँ?",
        ),
    ],
}
_LABEL_RE = re.compile(
    r"^\s*(hinglish|hindi|translation|converted text|output|answer)"
    r"(\s+(version|text|translation))?\s*[:：-]\s*",
    re.I,
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_USER_RE = re.compile(r"Convert this (\w+) text to (Hinglish|Hindi):\n\n(.*)\Z", re.S)


def user_message(text: str, source_lang: str, target: str) -> str:
    return f"Convert this {LANG_NAMES[source_lang]} text to {LANG_NAMES[target]}:\n\n{text}"


def clean_output(text: str, source: str) -> str:
    """Strip thinking blocks, a leading label and wrapping quotes; keep one paragraph when the source has
    one, and collapse whitespace when the source is a single line."""
    t = _THINK_RE.sub("", text or "").strip()
    t = _LABEL_RE.sub("", t, count=1).strip()
    if "\n\n" not in source.strip():
        t = t.split("\n\n")[0].strip()
    while len(t) >= 2 and t[0] in "\"'`“‘" and t[-1] in "\"'`”’":
        t = t[1:-1].strip()
    if "\n" not in source.strip():
        t = normalize_text(t)
    return t


# ── the generation tasks ─────────────────────────────────────────────────────


class _ConvertTask(BaseGenerationTask):
    """Shared prompt building, output gates and reject bookkeeping. Subclasses set ``target``,
    ``system_prompt`` and ``script_check``."""

    target = ""
    system_prompt = ""

    def __init__(
        self,
        llm: BaseLLM,
        *,
        ratio_bounds: tuple[float, float] = (0.5, 2.0),
        log_raw: bool = False,
        concurrency: int = 8,
        max_parse_retries: int = 1,
    ) -> None:
        super().__init__(llm, prompt_template=None, concurrency=concurrency, max_parse_retries=max_parse_retries)
        self.ratio_bounds = ratio_bounds
        self.log_raw = log_raw
        self.last_reason: dict[str, str] = {}  # sample id -> reason of its latest rejected attempt
        self.last_output: dict[str, str] = {}  # sample id -> latest rejected text (only if log_raw)
        self.attempts: dict[str, list[dict]] = {}  # sample id -> [{reason, chars, sha256}]

    @staticmethod
    def source_lang(sample: DataSample) -> str:
        return sample.metadata.get("lang") or ("hi" if contains_devanagari(sample.instruction) else "en")

    def _build_messages(self, sample: DataSample) -> list[dict[str, str]]:
        msgs = [{"role": "system", "content": self.system_prompt}]
        for lang, src, out in FEWSHOT[self.target]:
            msgs.append({"role": "user", "content": user_message(src, lang, self.target)})
            msgs.append({"role": "assistant", "content": out})
        msgs.append(
            {"role": "user", "content": user_message(sample.instruction, self.source_lang(sample), self.target)}
        )
        return msgs

    def script_check(self, text: str) -> str | None:
        raise NotImplementedError

    def check(self, source: str, text: str) -> str | None:
        """Reason to reject ``text`` as a conversion of ``source``, or None if it passes."""
        if not text:
            return "empty"
        bad = self.script_check(text)
        if bad:
            return bad
        if looks_like_refusal(text, source):
            return "refusal"
        ratio = len(normalize_text(text)) / max(len(normalize_text(source)), 1)
        lo, hi = self.ratio_bounds
        if not lo <= ratio <= hi:
            return f"length_ratio:{ratio:.2f}"
        return None

    def _parse_response(self, sample: DataSample, response) -> list[DataSample]:
        text = clean_output(getattr(response, "text", "") or "", sample.instruction)
        reason = self.check(sample.instruction, text)
        if reason:
            self.last_reason[sample.id] = reason
            self.attempts.setdefault(sample.id, []).append(
                {"reason": reason, "chars": len(text), "sha256": hashlib.sha256(text.encode()).hexdigest()[:16]}
            )
            if self.log_raw:
                self.last_output[sample.id] = text[:300]
            return []
        self.last_reason.pop(sample.id, None)
        meta = dict(sample.metadata)
        meta.update(
            lang=self.target,
            language=LANG_NAMES[self.target],
            source_id=sample.id,
            source_lang=self.source_lang(sample),
            converter_model=self.llm.model,
            converter_task=self.task_name,
        )
        return [
            DataSample(
                id=f"{sample.id}-{self.target}",
                source_uri=sample.source_uri,
                instruction=text,
                task_type="prompt_only",
                metadata=meta,
            )
        ]


class HinglishTask(_ConvertTask):
    """English or Hindi prompt -> romanised code-mixed Hinglish."""

    target = "hinglish"
    system_prompt = SYSTEM_HINGLISH

    def __init__(self, llm: BaseLLM, *, require_hinglish: bool = True, **kw) -> None:
        super().__init__(llm, **kw)
        self.require_hinglish = require_hinglish

    def script_check(self, text: str) -> str | None:
        return "devanagari" if contains_devanagari(text) else None

    def check(self, source: str, text: str) -> str | None:
        reason = super().check(source, text)
        if reason is None and self.require_hinglish and not looks_hinglish(text):
            return "not_hinglish"
        return reason


class HindiTranslationTask(_ConvertTask):
    """English prompt -> Hindi in Devanagari (for the XSTest safe prompts)."""

    target = "hi"
    system_prompt = SYSTEM_HINDI

    def script_check(self, text: str) -> str | None:
        return "not_devanagari" if devanagari_fraction(text) < 0.5 else None


TASKS = {"hinglish": HinglishTask, "hi": HindiTranslationTask}


# ── fake backend for --dry-run and tests ─────────────────────────────────────

_EN2HINGLISH = {
    "how": "kaise",
    "what": "kya",
    "why": "kyun",
    "where": "kahan",
    "when": "kab",
    "who": "kaun",
    "is": "hai",
    "are": "hain",
    "i": "main",
    "you": "aap",
    "me": "mujhe",
    "my": "mera",
    "your": "aapka",
    "and": "aur",
    "or": "ya",
    "not": "nahi",
    "in": "mein",
    "of": "ka",
    "for": "ke liye",
    "with": "ke saath",
    "to": "ko",
    "from": "se",
    "can": "sakta",
    "should": "chahiye",
    "tell": "batao",
    "make": "banao",
    "the": "",
    "a": "ek",
    "an": "ek",
    "this": "yeh",
    "that": "woh",
    "do": "karo",
    "please": "please",
}
_DEV_VOWELS = {
    "अ": "a",
    "आ": "aa",
    "इ": "i",
    "ई": "ee",
    "उ": "u",
    "ऊ": "oo",
    "ऋ": "ri",
    "ए": "e",
    "ऐ": "ai",
    "ओ": "o",
    "औ": "au",
}
_DEV_MATRAS = {
    "ा": "aa",
    "ि": "i",
    "ी": "ee",
    "ु": "u",
    "ू": "oo",
    "ृ": "ri",
    "े": "e",
    "ै": "ai",
    "ो": "o",
    "ौ": "au",
    "ॅ": "e",
    "ॉ": "o",
}
_DEV_CONS = dict(
    zip(
        "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळ",
        [
            "k",
            "kh",
            "g",
            "gh",
            "n",
            "ch",
            "chh",
            "j",
            "jh",
            "n",
            "t",
            "th",
            "d",
            "dh",
            "n",
            "t",
            "th",
            "d",
            "dh",
            "n",
            "p",
            "ph",
            "b",
            "bh",
            "m",
            "y",
            "r",
            "l",
            "v",
            "sh",
            "sh",
            "s",
            "h",
            "l",
        ],
    )
)
_DEV_OTHER = {"ं": "n", "ँ": "n", "ः": "h", "़": "", "्": "", "।": ".", "॥": "."}
_LAT2DEV = dict(zip("abcdefghijklmnopqrstuvwxyz", "अबकदएफगहइजकलमनओपकरसटउववकयज"))


def romanize_devanagari(text: str) -> str:
    """Crude Devanagari -> Latin transliteration (good enough for a fake backend)."""
    out: list[str] = []
    for k, c in enumerate(text):
        if c in _DEV_CONS:
            out.append(_DEV_CONS[c])
            nxt = text[k + 1 : k + 2]
            if nxt not in _DEV_MATRAS and nxt not in ("्", "़"):  # inherent vowel
                out.append("a")
        elif c in _DEV_MATRAS:
            out.append(_DEV_MATRAS[c])
        elif c in _DEV_VOWELS:
            out.append(_DEV_VOWELS[c])
        elif c in _DEV_OTHER:
            out.append(_DEV_OTHER[c])
        elif "०" <= c <= "९":
            out.append(str(ord(c) - 0x966))
        else:
            out.append(c)
    return "".join(out)


class FakeLLM(BaseLLM):
    """Offline stand-in. By default: Hindi -> crude romanisation, English -> word-swap Hinglish, and any
    text -> "Hindi" as letter-by-letter Devanagari. ``script(source_text, target) -> reply`` overrides
    that, e.g. to inject refusals."""

    def __init__(self, script=None) -> None:
        super().__init__(model="fake/dry-run", temperature=0.0, max_tokens=512, max_retries=0)
        self.script = script
        self.calls = 0

    def _call(self, messages, **kwargs) -> LLMResponse:
        self.calls += 1
        m = _USER_RE.search(messages[-1]["content"])
        src, target = (m.group(3), m.group(2).lower()) if m else (messages[-1]["content"], "hinglish")
        target = "hi" if target == "hindi" else target
        reply = self.script(src, target) if self.script else self.default(src, target)
        return LLMResponse(text=reply, model=self.model)

    @staticmethod
    def default(src: str, target: str) -> str:
        if target == "hi":
            return "".join(_LAT2DEV.get(c.lower(), c) for c in src)
        if contains_devanagari(src):
            return romanize_devanagari(src)
        words = [_EN2HINGLISH.get(w.strip(".,?!;:").lower(), w) for w in src.split()]
        out = " ".join(w for w in words if w)
        return out if looks_hinglish(out) else out.rstrip("?.!") + " batao?"


def make_llm(
    model: str | None, api_base: str | None, *, dry_run: bool, temperature: float = 0.3, max_tokens: int = 512
) -> BaseLLM:
    if dry_run:
        return FakeLLM()
    if not model:
        raise SystemExit("--model is required unless --dry-run (e.g. openai/CohereLabs/aya-expanse-8b)")
    from curatorkit.llm.litellm import LiteLLMBackend

    from pencil.config import load_env

    load_env()
    return LiteLLMBackend(model=model, api_base=api_base, temperature=temperature, max_tokens=max_tokens)


# ── running a conversion over a set ──────────────────────────────────────────


def _row_to_sample(row: dict, *, path: str) -> DataSample:
    return DataSample(
        id=str(row["id"]),
        source_uri=f"file://{path}",
        instruction=str(row["prompt"]),
        task_type="prompt_only",
        metadata=dict(row.get("metadata") or {}),
    )


def run_split(task: _ConvertTask, inputs: list[Path], work_dir: Path):
    """One CuratorKIT Pipeline run: a JSONLReader per input file -> task -> PromptsMetaExporter.
    Writes manifest.json + lexsi_provenance.json (no rejected sidecar: it holds raw LLM text)."""
    from curatorkit.connectors.jsonl import JSONLReader
    from curatorkit.exporters.prompts_meta import PromptsMetaExporter
    from curatorkit.manifest import ProvenanceManifest, build_provenance, dataset_input, step_models
    from curatorkit.pipeline import Pipeline

    readers = [
        JSONLReader(path=p, format="prompt_only", preprocessing_fn=partial(_row_to_sample, path=str(p))) for p in inputs
    ]
    steps = readers + [task, PromptsMetaExporter()]
    result = asyncio.run(Pipeline(steps, output_dir=work_dir).run_async())
    provenance = build_provenance(
        method=f"pencil.{task.task_name}",
        inputs=[dataset_input(str(p)) for p in inputs],
        params={**step_models(steps), "task_config_hash": task._config_hash(), "ratio_bounds": list(task.ratio_bounds)},
    )
    ProvenanceManifest(result, task._config_hash(), work_dir, provenance=provenance).write()
    return result


def _read_sources(inputs: list[Path]) -> list[dict]:
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs {missing}; build them first")
    return [r for p in inputs for r in read_jsonl(p)]


def _passed_rows(result, source_rows: list[dict], split: str) -> list[dict]:
    """The converted rows in the order of their sources, tagged with their split."""
    order = {r["id"]: k for k, r in enumerate(source_rows)}
    rows = [{"id": s.id, "prompt": s.instruction, "metadata": {**s.metadata, "split": split}} for s in result.passed]
    return sorted(rows, key=lambda r: order[r["metadata"]["source_id"]])


def _rejected_entries(result, task: _ConvertTask, where: dict) -> list[dict]:
    """One rejects-log entry per rejected sample: its id, reason and attempts (lengths and hashes only,
    plus a text preview when the task keeps them)."""
    entries = []
    for r in result.rejected:
        entry = {
            **where,
            "id": r.id,
            "reason": task.last_reason.get(r.id) or r.rejection_reason,
            "attempts": task.attempts.get(r.id, []),
        }
        if r.id in task.last_output:
            entry["output_preview"] = task.last_output[r.id]
        entries.append(entry)
    return entries


def _drop_repeated_outputs(converted: dict[str, list[dict]]) -> dict[str, list[str]]:
    """An en and a hi prompt with the same meaning can convert to the same text. Keep each normalised
    output once, visiting test, then dev, then factor; returns the dropped ids per split."""
    seen: set[str] = set()
    dropped: dict[str, list[str]] = {}
    for split in sorted(converted, key=SPLIT_KEEP_ORDER.index):
        kept, dropped[split] = [], []
        for r in converted[split]:
            key = normalize_for_match(r["prompt"])
            if key in seen:
                dropped[split].append(r["id"])
            else:
                seen.add(key)
                kept.append(r)
        converted[split] = kept
    return dropped


def _review_sample(rows: list[dict], frac: float, seed: str) -> list[dict]:
    """A seeded random ``frac`` of rows, at least one, for the human spot check (none when frac is 0)."""
    k = min(len(rows), max(1, math.ceil(frac * len(rows)))) if rows and frac > 0 else 0
    return random.Random(seed).sample(rows, k)


def _review_row(row: dict, set_name: str, source_prompts: dict[str, str], target: str) -> dict:
    """One spot-check CSV row: the source prompt beside its conversion, with blank verdict columns."""
    meta = row["metadata"]
    return {
        "set": set_name,
        "split": meta["split"],
        "id": row["id"],
        "source_id": meta["source_id"],
        "source_lang": meta["source_lang"],
        "source_prompt": source_prompts[meta["source_id"]],
        target: row["prompt"],
        "ok (y/n)": "",
        "notes": "",
    }


def _write_review_csv(path: Path, rows: list[dict], target: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        fields = ["set", "split", "id", "source_id", "source_lang", "source_prompt", target, "ok (y/n)", "notes"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run_conversion(
    *,
    set_name: str,
    target: str,
    llm: BaseLLM,
    data_dir: Path,
    out_dir: Path,
    splits=None,
    sources=None,
    review_dir: Path = DEFAULT_REVIEW_DIR,
    review_frac: float = 0.1,
    seed: int = 0,
    ratio_bounds=(0.5, 2.0),
    require_hinglish: bool = True,
    concurrency: int = 8,
    max_parse_retries: int = 2,
    log_raw: bool = False,
) -> dict:
    """Convert ``{set}/{source}/{split}.jsonl`` into ``{set}/{target}/{split}.jsonl`` under ``out_dir`` for
    every split. Also writes the review CSV, the rejects log and conversion.json; returns that summary."""
    splits = list(splits or PROMPT_SET_SPLITS[set_name])
    sources = list(sources or SET_SOURCES[set_name])
    log_raw = log_raw and set_name != "harmful"  # never keep raw text produced from harmful prompts
    task_kw = dict(
        ratio_bounds=tuple(ratio_bounds), log_raw=log_raw, concurrency=concurrency, max_parse_retries=max_parse_retries
    )
    if target == "hinglish":
        task_kw["require_hinglish"] = require_hinglish
    out_set = out_dir / set_name / target
    summary = {"set": set_name, "target": target, "sources": sources, "model": llm.model, "splits": {}}
    converted, rejects, source_prompts = {}, [], {}
    for split in splits:
        inputs = [data_dir / set_name / src / f"{split}.jsonl" for src in sources]
        source_rows = _read_sources(inputs)
        source_prompts.update({r["id"]: r["prompt"] for r in source_rows})
        task = TASKS[target](llm, **task_kw)
        result = run_split(task, inputs, out_set / "_curatorkit" / split)
        converted[split] = _passed_rows(result, source_rows, split)
        split_rejects = _rejected_entries(result, task, {"set": set_name, "target": target, "split": split})
        rejects += split_rejects
        reasons = Counter(e["reason"].split(":")[0] for e in split_rejects)
        summary["splits"][split] = {
            "n_source": len(source_rows),
            "n_rejected": len(result.rejected),
            "rejected_by_reason": dict(reasons),
        }

    for split, ids in _drop_repeated_outputs(converted).items():
        rejects += [
            {"set": set_name, "target": target, "split": split, "id": i, "reason": "duplicate_output", "attempts": []}
            for i in ids
        ]
        summary["splits"][split]["n_duplicate_output"] = len(ids)
    if len(splits) < len(PROMPT_SET_SPLITS[set_name]):
        print(f"[{target}] note: duplicate outputs were only checked across {splits}")

    review_rows = []
    for split in splits:
        rows = converted[split]
        write_jsonl(out_set / f"{split}.jsonl", rows)
        sample = _review_sample(rows, review_frac, f"{seed}-{set_name}-{target}-{split}")
        review_rows += [_review_row(r, set_name, source_prompts, target) for r in sample]
        s = summary["splits"][split]
        s.update(n_out=len(rows), n_review=len(sample))
        print(
            f"[{target}] {set_name}/{split}: {len(rows)}/{s['n_source']} kept, {s['n_rejected']} rejected "
            f"{s['rejected_by_reason'] or ''}, {s['n_duplicate_output']} duplicate outputs dropped"
        )

    suffix = "" if set_name == "harmful" else f"_{set_name}"
    review_path = review_dir / f"{target}_check{suffix}.csv"
    rejects_path = review_dir / f"{target}_rejects{suffix}.jsonl"
    _write_review_csv(review_path, review_rows, target)
    write_jsonl(rejects_path, rejects)
    summary.update(review_csv=str(review_path), rejects_log=str(rejects_path), created=timestamp())
    write_manifest(out_set / "conversion.json", summary)
    print(f"[{target}] review sample -> {review_path}; rejects -> {rejects_path}")
    return summary


def add_llm_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("LLM")
    g.add_argument(
        "--model",
        default=None,
        help="LiteLLM model, e.g. openai/CohereLabs/aya-expanse-8b or cohere_chat/c4ai-aya-expanse-32b",
    )
    g.add_argument("--api-base", default=None, help="e.g. http://localhost:8000/v1 for a local vLLM server")
    g.add_argument("--temperature", type=float, default=0.3, help="> 0 so a retry can differ")
    g.add_argument("--max-tokens", type=int, default=512)
    g.add_argument("--concurrency", type=int, default=8)
    g.add_argument("--max-parse-retries", type=int, default=2, help="extra attempts after a rejected output")
    g.add_argument("--ratio-min", type=float, default=0.5)
    g.add_argument("--ratio-max", type=float, default=2.0)
    g.add_argument(
        "--no-require-hinglish",
        action="store_true",
        help="do not reject outputs without romanised Hindi function words",
    )
    g.add_argument("--dry-run", action="store_true", help="fake offline LLM; writes under outputs/dry_run/")
    g.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    g.add_argument("--review-frac", type=float, default=0.1)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__
    )
    p.add_argument("--set", dest="set_name", choices=sorted(PROMPT_SET_SPLITS), default="harmful")
    p.add_argument("--splits", nargs="+", default=None, help="default: every split of the set")
    p.add_argument(
        "--sources",
        nargs="+",
        choices=["en", "hi"],
        default=None,
        help="source languages (default: en hi for harmful, en for xstest)",
    )
    p.add_argument("--data-dir", default=None, help="processed data root to read (default: data/processed)")
    p.add_argument(
        "--out-dir", default=None, help="root to write (default: --data-dir; outputs/dry_run with --dry-run)"
    )
    p.add_argument("--seed", type=int, default=0, help="seed for the review sample")
    p.add_argument(
        "--log-raw-rejects",
        action="store_true",
        help="keep a 300-char preview of rejected outputs (xstest only; ignored for harmful)",
    )
    add_llm_args(p)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    data_dir = resolve_data_dir(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (DRY_RUN_DIR if args.dry_run else data_dir)
    if args.log_raw_rejects and args.set_name == "harmful":
        print("[hinglish] --log-raw-rejects is ignored for the harmful set")
    llm = make_llm(
        args.model, args.api_base, dry_run=args.dry_run, temperature=args.temperature, max_tokens=args.max_tokens
    )
    run_conversion(
        set_name=args.set_name,
        target="hinglish",
        llm=llm,
        data_dir=data_dir,
        out_dir=out_dir,
        splits=args.splits,
        sources=args.sources,
        review_dir=Path(args.review_dir),
        review_frac=args.review_frac,
        seed=args.seed,
        ratio_bounds=(args.ratio_min, args.ratio_max),
        require_hinglish=not args.no_require_hinglish,
        concurrency=args.concurrency,
        max_parse_retries=args.max_parse_retries,
        log_raw=args.log_raw_rejects,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
