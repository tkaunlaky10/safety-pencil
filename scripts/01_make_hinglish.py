#!/usr/bin/env python
"""Make Hinglish versions of prompt sets through an LLM, as a CuratorKIT generation task.

Hinglish here means romanised, code-mixed Hindi-English: Latin script only, meaning preserved,
nothing added or removed. Each English and Hindi prompt of a split becomes one Hinglish prompt
of the *same* split, with id ``<source_id>-hinglish``:

    harmful/{en,hi}/{split}.jsonl  ->  harmful/hinglish/{split}.jsonl      (--set harmful)
    xstest/en/{split}.jsonl        ->  xstest/hinglish/{split}.jsonl       (--set xstest)

``HinglishTask`` (and ``HindiTranslationTask``, used by 01_build_xstest.py for en -> hi) subclass
CuratorKIT's ``BaseGenerationTask`` and run through ``Pipeline`` (JSONLReader -> task ->
PromptsMetaExporter). Outputs are rejected when they still contain Devanagari, look like a refusal
("I can't", "माफ़", "maaf kijiye", ...), have a character-length ratio to the source outside
[0.5, 2.0], or do not look like Hinglish (no romanised Hindi function words). Rejects are logged
by id and reason; for the harmful set the rejected text itself is never written anywhere (it could
be an answer to the harmful prompt), only its length and hash.

A random 10% of each split goes to outputs/review/hinglish_check.csv for a human spot check
(outputs/ is gitignored). ``--dry-run`` swaps in a fake offline LLM and, unless --out-dir is
given, writes under outputs/dry_run/ so real data is never overwritten.

LLM: any LiteLLM model string, e.g. ``--model openai/CohereLabs/aya-expanse-8b --api-base
http://localhost:8000/v1`` (local vLLM) or ``--model cohere_chat/c4ai-aya-expanse-32b``
(COHERE_API_KEY in .env).
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from curatorkit.generators.base import BaseGenerationTask  # noqa: E402
from curatorkit.llm.base import BaseLLM, LLMResponse  # noqa: E402
from curatorkit.schema import DataSample  # noqa: E402

from pencil.datasets import (  # noqa: E402
    HARMFUL_SPLITS, REPO_ROOT, XSTEST_SPLITS, contains_devanagari, devanagari_fraction,
    looks_hinglish, looks_like_refusal, normalize_for_match, normalize_text, read_jsonl,
    resolve_data_dir, write_jsonl,
)

LANG_NAMES = {"en": "English", "hi": "Hindi", "hinglish": "Hinglish"}
SET_SPLITS = {"harmful": HARMFUL_SPLITS, "xstest": XSTEST_SPLITS}
SET_SOURCES = {"harmful": ("en", "hi"), "xstest": ("en",)}
SPLIT_KEEP_ORDER = ("test", "dev", "factor")      # who keeps a duplicated output
DEFAULT_REVIEW_DIR = REPO_ROOT / "outputs" / "review"

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
        ("en", "How can I fix my sleep schedule before the exams start next week?",
         "Next week exams start ho rahe hain, usse pehle main apna sleep schedule kaise theek karun?"),
        ("hi", "मुझे बताइए कि घर पर अच्छी मसाला चाय कैसे बनाते हैं।",
         "Mujhe batao ki ghar par acchi masala chai kaise banate hain."),
    ],
    "hi": [
        ("en", "How can I fix my sleep schedule before the exams start next week?",
         "अगले हफ़्ते परीक्षा शुरू होने से पहले मैं अपना सोने का समय कैसे ठीक करूँ?"),
    ],
}
_LABEL_RE = re.compile(r"^\s*(hinglish|hindi|translation|converted text|output|answer)"
                       r"(\s+(version|text|translation))?\s*[:：-]\s*", re.I)
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_USER_RE = re.compile(r"Convert this (\w+) text to (Hinglish|Hindi):\n\n(.*)\Z", re.S)


def user_message(text: str, source_lang: str, target: str) -> str:
    return f"Convert this {LANG_NAMES[source_lang]} text to {LANG_NAMES[target]}:\n\n{text}"


def clean_output(text: str, source: str) -> str:
    """Strip thinking blocks, a leading label and wrapping quotes; keep one paragraph when the
    source has one, and collapse whitespace when the source is a single line."""
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
    """Shared gates and bookkeeping. Subclasses set ``target`` and ``system_prompt``."""

    target = ""
    system_prompt = ""

    def __init__(self, llm: BaseLLM, *, ratio_bounds: tuple[float, float] = (0.5, 2.0),
                 log_raw: bool = False, concurrency: int = 8, max_parse_retries: int = 1) -> None:
        super().__init__(llm, prompt_template=None, concurrency=concurrency,
                         max_parse_retries=max_parse_retries)
        self.ratio_bounds = ratio_bounds
        self.log_raw = log_raw
        self.last_reason: dict[str, str] = {}      # sample id -> reason of its latest rejected attempt
        self.last_output: dict[str, str] = {}      # sample id -> latest rejected text (only if log_raw)
        self.attempts: dict[str, list[dict]] = {}  # sample id -> [{reason, chars, sha256}]

    @staticmethod
    def source_lang(sample: DataSample) -> str:
        return sample.metadata.get("lang") or ("hi" if contains_devanagari(sample.instruction) else "en")

    def _build_messages(self, sample: DataSample) -> list[dict[str, str]]:
        msgs = [{"role": "system", "content": self.system_prompt}]
        for lang, src, out in FEWSHOT[self.target]:
            msgs.append({"role": "user", "content": user_message(src, lang, self.target)})
            msgs.append({"role": "assistant", "content": out})
        msgs.append({"role": "user", "content": user_message(sample.instruction, self.source_lang(sample),
                                                             self.target)})
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
                {"reason": reason, "chars": len(text),
                 "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]})
            if self.log_raw:
                self.last_output[sample.id] = text[:300]
            return []
        self.last_reason.pop(sample.id, None)
        meta = dict(sample.metadata)
        meta.update(lang=self.target, language=LANG_NAMES[self.target], source_id=sample.id,
                    source_lang=self.source_lang(sample), converter_model=self.llm.model,
                    converter_task=self.task_name)
        return [DataSample(id=f"{sample.id}-{self.target}", source_uri=sample.source_uri, instruction=text,
                           task_type="prompt_only", metadata=meta)]


class HinglishTask(_ConvertTask):
    """English or Hindi prompt -> romanised code-mixed Hinglish."""

    target = "hinglish"
    system_prompt = SYSTEM_HINGLISH

    def __init__(self, llm: BaseLLM, *, require_hinglish: bool = True, **kw) -> None:
        super().__init__(llm, **kw)
        self.require_hinglish = require_hinglish

    def script_check(self, text: str) -> str | None:
        if contains_devanagari(text):
            return "devanagari"
        return None

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
        if devanagari_fraction(text) < 0.5:
            return "not_devanagari"
        return None


TASKS = {"hinglish": HinglishTask, "hi": HindiTranslationTask}


# ── fake backend for --dry-run and tests ─────────────────────────────────────

_EN2HINGLISH = {
    "how": "kaise", "what": "kya", "why": "kyun", "where": "kahan", "when": "kab", "who": "kaun",
    "is": "hai", "are": "hain", "i": "main", "you": "aap", "me": "mujhe", "my": "mera", "your": "aapka",
    "and": "aur", "or": "ya", "not": "nahi", "in": "mein", "of": "ka", "for": "ke liye", "with": "ke saath",
    "to": "ko", "from": "se", "can": "sakta", "should": "chahiye", "tell": "batao", "make": "banao",
    "the": "", "a": "ek", "an": "ek", "this": "yeh", "that": "woh", "do": "karo", "please": "please",
}
_DEV_VOWELS = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri", "ए": "e",
               "ऐ": "ai", "ओ": "o", "औ": "au"}
_DEV_MATRAS = {"ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri", "े": "e", "ै": "ai",
               "ो": "o", "ौ": "au", "ॅ": "e", "ॉ": "o"}
_DEV_CONS = dict(zip("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळ",
                     "k kh g gh n ch chh j jh n t th d dh n t th d dh n p ph b bh m y r l v sh sh s h l"
                     .split()))
_DEV_OTHER = {"ं": "n", "ँ": "n", "ः": "h", "़": "", "्": "", "।": ".", "॥": "."}
_LAT2DEV = dict(zip("abcdefghijklmnopqrstuvwxyz", "अबकदएफगहइजकलमनओपकरसटउववकयज"))


def romanize_devanagari(text: str) -> str:
    """Crude Devanagari -> Latin transliteration (good enough for a fake backend)."""
    out: list[str] = []
    chars = list(text)
    for k, c in enumerate(chars):
        nxt = chars[k + 1] if k + 1 < len(chars) else ""
        if c in _DEV_CONS:
            out.append(_DEV_CONS[c])
            if nxt not in _DEV_MATRAS and nxt != "्" and nxt != "़":
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
    """Offline stand-in. Default behaviour: Hindi -> crude romanisation, English -> word-swap
    Hinglish, anything -> Hindi as letter-by-letter Devanagari. ``script`` (callable taking
    (source_text, target) and returning the reply) overrides it, e.g. to inject refusals."""

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
        words = []
        for w in src.split():
            core = w.strip(".,?!;:").lower()
            words.append(_EN2HINGLISH.get(core, w) if core in _EN2HINGLISH else w)
        out = " ".join(w for w in words if w)
        return out if looks_hinglish(out) else out.rstrip("?.!") + " batao?"


def make_llm(model: str | None, api_base: str | None, *, dry_run: bool, temperature: float = 0.3,
             max_tokens: int = 512) -> BaseLLM:
    if dry_run:
        return FakeLLM()
    if not model:
        raise SystemExit("--model is required unless --dry-run (e.g. openai/CohereLabs/aya-expanse-8b)")
    from pencil.config import load_env
    from curatorkit.llm.litellm import LiteLLMBackend

    load_env()
    return LiteLLMBackend(model=model, api_base=api_base, temperature=temperature, max_tokens=max_tokens)


# ── running a conversion over a set ──────────────────────────────────────────

def _row_to_sample(row: dict, *, path: str) -> DataSample:
    return DataSample(id=str(row["id"]), source_uri=f"file://{path}", instruction=str(row["prompt"]),
                      task_type="prompt_only", metadata=dict(row.get("metadata") or {}))


def run_split(task: _ConvertTask, inputs: list[Path], work_dir: Path):
    """One CuratorKIT Pipeline run: a JSONLReader per input file -> task -> PromptsMetaExporter.
    Writes manifest.json + lexsi_provenance.json (no rejected sidecar: it holds raw LLM text)."""
    from curatorkit.connectors.jsonl import JSONLReader
    from curatorkit.exporters.prompts_meta import PromptsMetaExporter
    from curatorkit.manifest import ProvenanceManifest, build_provenance, dataset_input, step_models
    from curatorkit.pipeline import Pipeline

    readers = [JSONLReader(path=p, format="prompt_only", preprocessing_fn=partial(_row_to_sample, path=str(p)))
               for p in inputs]
    steps = readers + [task, PromptsMetaExporter()]
    result = asyncio.run(Pipeline(steps, output_dir=work_dir).run_async())
    provenance = build_provenance(
        method=f"pencil.{task.task_name}",
        inputs=[dataset_input(str(p)) for p in inputs],
        params={**step_models(steps), "task_config_hash": task._config_hash(),
                "ratio_bounds": list(task.ratio_bounds)},
    )
    ProvenanceManifest(result, task._config_hash(), work_dir, provenance=provenance).write()
    return result


def run_conversion(*, set_name: str, target: str, llm: BaseLLM, data_dir: Path, out_dir: Path,
                   splits=None, sources=None, review_dir: Path = DEFAULT_REVIEW_DIR,
                   review_frac: float = 0.1, seed: int = 0, ratio_bounds=(0.5, 2.0),
                   require_hinglish: bool = True, concurrency: int = 8, max_parse_retries: int = 2,
                   log_raw: bool = False) -> dict:
    """Convert ``{set}/{source}/{split}.jsonl`` for every split into ``{set}/{target}/{split}.jsonl``
    under ``out_dir``. Returns a summary; also writes the review CSV and the rejects log."""
    splits = list(splits or SET_SPLITS[set_name])
    sources = list(sources or SET_SOURCES[set_name])
    log_raw = log_raw and set_name != "harmful"       # never keep raw text produced from harmful prompts
    kw = dict(ratio_bounds=tuple(ratio_bounds), log_raw=log_raw, concurrency=concurrency,
              max_parse_retries=max_parse_retries)
    if target == "hinglish":
        kw["require_hinglish"] = require_hinglish
    suffix = "" if set_name == "harmful" else f"_{set_name}"
    review_rows, reject_rows, summary = [], [], {"set": set_name, "target": target, "sources": sources,
                                                 "model": llm.model, "splits": {}}
    converted, prompts = {}, {}
    for split in splits:
        inputs = [data_dir / set_name / src / f"{split}.jsonl" for src in sources]
        missing = [str(p) for p in inputs if not p.exists()]
        if missing:
            raise FileNotFoundError(f"missing inputs {missing}; build them first")
        source_rows = [r for p in inputs for r in read_jsonl(p)]
        order = {r["id"]: k for k, r in enumerate(source_rows)}
        prompts.update({r["id"]: r["prompt"] for r in source_rows})
        task = TASKS[target](llm, **kw)
        result = run_split(task, inputs, out_dir / set_name / target / "_curatorkit" / split)
        rows = sorted(({"id": s.id, "prompt": s.instruction, "metadata": {**s.metadata, "split": split}}
                       for s in result.passed), key=lambda r: order.get(r["metadata"]["source_id"], 1 << 30))
        reasons: dict[str, int] = {}
        for r in result.rejected:
            reason = task.last_reason.get(r.id) or r.rejection_reason
            reasons[reason.split(":")[0]] = reasons.get(reason.split(":")[0], 0) + 1
            entry = {"set": set_name, "target": target, "split": split, "id": r.id, "reason": reason,
                     "attempts": task.attempts.get(r.id, [])}
            if log_raw and r.id in task.last_output:
                entry["output_preview"] = task.last_output[r.id]
            reject_rows.append(entry)
        converted[split] = rows
        summary["splits"][split] = {"n_source": len(source_rows), "n_rejected": len(result.rejected),
                                    "rejected_by_reason": reasons}

    # An en prompt and a hi prompt with the same meaning can convert to the same text. Drop exact
    # (normalised) repeats within a split, and across splits keep test first, then dev, then factor.
    seen: set[str] = set()
    for split in sorted(converted, key=lambda s: (SPLIT_KEEP_ORDER.index(s) if s in SPLIT_KEEP_ORDER else 9, s)):
        kept = []
        for r in converted[split]:
            key = normalize_for_match(r["prompt"])
            if key in seen:
                reject_rows.append({"set": set_name, "target": target, "split": split, "id": r["id"],
                                    "reason": "duplicate_output", "attempts": []})
                continue
            seen.add(key)
            kept.append(r)
        summary["splits"][split]["n_duplicate_output"] = len(converted[split]) - len(kept)
        converted[split] = kept
    if len(splits) < len(SET_SPLITS[set_name]):
        print(f"[{target}] note: duplicate outputs were only checked across {splits}")

    for split in splits:
        rows = converted[split]
        write_jsonl(out_dir / set_name / target / f"{split}.jsonl", rows)
        k = min(len(rows), max(1, math.ceil(review_frac * len(rows)))) if rows and review_frac > 0 else 0
        for r in random.Random(f"{seed}-{set_name}-{target}-{split}").sample(rows, k):
            src_id = r["metadata"]["source_id"]
            review_rows.append({"set": set_name, "split": split, "id": r["id"], "source_id": src_id,
                                "source_lang": r["metadata"]["source_lang"], "source_prompt": prompts[src_id],
                                target: r["prompt"], "ok (y/n)": "", "notes": ""})
        s = summary["splits"][split]
        s.update(n_out=len(rows), n_review=k)
        print(f"[{target}] {set_name}/{split}: {len(rows)}/{s['n_source']} kept, {s['n_rejected']} rejected "
              f"{s['rejected_by_reason'] or ''}, {s['n_duplicate_output']} duplicate outputs dropped")

    review_dir.mkdir(parents=True, exist_ok=True)
    review_path = review_dir / f"{target}_check{suffix}.csv"
    with open(review_path, "w", newline="", encoding="utf-8-sig") as f:
        fields = ["set", "split", "id", "source_id", "source_lang", "source_prompt", target, "ok (y/n)", "notes"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(review_rows)
    rejects_path = review_dir / f"{target}_rejects{suffix}.jsonl"
    write_jsonl(rejects_path, reject_rows)
    summary.update(review_csv=str(review_path), rejects_log=str(rejects_path),
                   created=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    (out_dir / set_name / target).mkdir(parents=True, exist_ok=True)
    (out_dir / set_name / target / "conversion.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"[{target}] review sample -> {review_path}; rejects -> {rejects_path}")
    return summary


def add_llm_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("LLM")
    g.add_argument("--model", default=None,
                   help="LiteLLM model, e.g. openai/CohereLabs/aya-expanse-8b or cohere_chat/c4ai-aya-expanse-32b")
    g.add_argument("--api-base", default=None, help="e.g. http://localhost:8000/v1 for a local vLLM server")
    g.add_argument("--temperature", type=float, default=0.3, help="> 0 so a retry can differ")
    g.add_argument("--max-tokens", type=int, default=512)
    g.add_argument("--concurrency", type=int, default=8)
    g.add_argument("--max-parse-retries", type=int, default=2, help="extra attempts after a rejected output")
    g.add_argument("--ratio-min", type=float, default=0.5)
    g.add_argument("--ratio-max", type=float, default=2.0)
    g.add_argument("--no-require-hinglish", action="store_true",
                   help="do not reject outputs without romanised Hindi function words")
    g.add_argument("--dry-run", action="store_true", help="fake offline LLM; writes under outputs/dry_run/")
    g.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    g.add_argument("--review-frac", type=float, default=0.1)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--set", dest="set_name", choices=sorted(SET_SPLITS), default="harmful")
    p.add_argument("--splits", nargs="+", default=None, help="default: every split of the set")
    p.add_argument("--sources", nargs="+", choices=["en", "hi"], default=None,
                   help="source languages (default: en hi for harmful, en for xstest)")
    p.add_argument("--data-dir", default=None, help="processed data root to read (default: data/processed)")
    p.add_argument("--out-dir", default=None, help="root to write (default: --data-dir; outputs/dry_run with --dry-run)")
    p.add_argument("--seed", type=int, default=0, help="seed for the review sample")
    p.add_argument("--log-raw-rejects", action="store_true",
                   help="keep a 300-char preview of rejected outputs (xstest only; ignored for harmful)")
    add_llm_args(p)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    data_dir = resolve_data_dir(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO_ROOT / "outputs" / "dry_run" / "processed" if args.dry_run else data_dir)
    if args.log_raw_rejects and args.set_name == "harmful":
        print("[hinglish] --log-raw-rejects is ignored for the harmful set")
    llm = make_llm(args.model, args.api_base, dry_run=args.dry_run, temperature=args.temperature,
                   max_tokens=args.max_tokens)
    run_conversion(set_name=args.set_name, target="hinglish", llm=llm, data_dir=data_dir, out_dir=out_dir,
                   splits=args.splits, sources=args.sources, review_dir=Path(args.review_dir),
                   review_frac=args.review_frac, seed=args.seed, ratio_bounds=(args.ratio_min, args.ratio_max),
                   require_hinglish=not args.no_require_hinglish, concurrency=args.concurrency,
                   max_parse_retries=args.max_parse_retries, log_raw=args.log_raw_rejects)
    return 0


if __name__ == "__main__":
    sys.exit(main())
