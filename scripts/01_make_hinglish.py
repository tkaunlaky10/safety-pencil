#!/usr/bin/env python
"""Make Hinglish versions of prompt sets through an LLM, as a CuratorKIT generation task.

Hinglish here means romanised, code-mixed Hindi-English: Latin script only, meaning preserved, nothing added
or removed.

What is converted. Each en and hi prompt of a split becomes one Hinglish prompt of the same split, with id
``<source_id>-hinglish``:

    harmful/{en,hi}/{split}.jsonl  ->  harmful/hinglish/{split}.jsonl      (--set harmful)
    xstest/en/{split}.jsonl        ->  xstest/hinglish/{split}.jsonl       (--set xstest)

``HinglishTask`` (and ``HindiTranslationTask``, which 01_build_xstest.py uses for en -> hi) run through
CuratorKIT's ``Pipeline`` (JSONLReader -> task -> PromptsMetaExporter). The LLM gets a system prompt and
benign few-shot examples; its reply is cleaned (``clean_output``) and then checked.

Which checks reject a conversion (``_ConvertTask.check``, in this order; the first failure is the reason, and
CuratorKIT asks again up to --max-parse-retries times):

    empty              nothing is left after cleaning
    devanagari         (Hinglish) any Devanagari character: the output must be romanised
    not_devanagari     (Hindi) less than half of the letters are Devanagari
    refusal            the reply looks like a refusal rather than a conversion
    length_ratio:R     the output/source length ratio R is outside [0.5, 2.0] (--ratio-min, --ratio-max),
                       so text was probably dropped or added
    not_hinglish       (Hinglish) no romanised Hindi function words (--no-require-hinglish turns this off)
    duplicate_output   checked after all splits: two prompts converted to the same text, which is then kept
                       once, preferring test, then dev, then factor

What gets written. Under --out-dir: ``{set}/hinglish/{split}.jsonl`` (the conversions that passed),
``{set}/hinglish/conversion.json`` (counts per split and reason) and ``{set}/hinglish/_curatorkit/{split}/``
(CuratorKIT's export, manifest and provenance). Under --review-dir (default outputs/review/, gitignored;
xstest file names end in _xstest): ``hinglish_check.csv``, a random 10% of each split for a human spot check,
and ``hinglish_rejects.jsonl``, each reject by id, reason, length and hash. For the harmful set the rejected
text itself is never written anywhere, since it could answer the harmful prompt.

``--dry-run`` uses a fake offline LLM and, unless --out-dir is given, writes under outputs/dry_run/ so real
data is never overwritten.

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
import os
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

DEFAULT_RATIO_BOUNDS = (0.5, 2.0)  # allowed output/source length ratio
MIN_DEVANAGARI_FRACTION = 0.5  # at least this share of a Hindi translation's letters must be Devanagari
DEFAULT_REVIEW_FRAC = 0.1  # share of each split drawn for the human spot check
PREVIEW_CHARS = 300  # --log-raw-rejects keeps this much of a rejected output (never for harmful prompts)
HASH_CHARS = 16  # rejects are logged by this many hex digits of their sha256
API_KEY_ENV = "OPENAI_API_KEY"  # LiteLLM's OpenAI-compatible key, also for the Hugging Face router
MAX_REJECTED_FRACTION = 0.5  # a split losing more than this to rejects means the LLM step is broken

# ── what the LLM is told ─────────────────────────────────────────────────────

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
# Benign worked examples per target: (source language, source text, expected output).
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


def user_message(text: str, source_lang: str, target: str) -> str:
    """The user turn asking for one conversion, e.g. "Convert this English text to Hinglish:\\n\\n<text>"."""
    return f"Convert this {LANG_NAMES[source_lang]} text to {LANG_NAMES[target]}:\n\n{text}"


# ── cleaning the reply ───────────────────────────────────────────────────────

# A leading label the model sometimes adds, e.g. "Hinglish:", "Translation -" or "Hindi version:".
_LABEL_RE = re.compile(
    r"^\s*(hinglish|hindi|translation|converted text|output|answer)"
    r"(\s+(version|text|translation))?\s*[:：-]\s*",
    re.I,
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)  # a reasoning model's thinking block
_OPENING_QUOTES = "\"'`“‘"
_CLOSING_QUOTES = "\"'`”’"


def _strip_wrapping_quotes(text: str) -> str:
    """text without the quote marks around it, removed pair by pair while it starts and ends with one."""
    while len(text) >= 2 and text[0] in _OPENING_QUOTES and text[-1] in _CLOSING_QUOTES:
        text = text[1:-1].strip()
    return text


def clean_output(text: str, source: str) -> str:
    """The reply without thinking blocks, a leading label and wrapping quotes; cut to its first paragraph when
    the source has one paragraph, and with whitespace collapsed when the source is a single line."""
    if not text:
        text = ""
    source_has_one_paragraph = "\n\n" not in source.strip()
    source_is_one_line = "\n" not in source.strip()

    cleaned = _THINK_RE.sub("", text).strip()
    cleaned = _LABEL_RE.sub("", cleaned, count=1).strip()
    if source_has_one_paragraph:
        cleaned = cleaned.split("\n\n")[0].strip()
    cleaned = _strip_wrapping_quotes(cleaned)
    if source_is_one_line:
        cleaned = normalize_text(cleaned)
    return cleaned


# ── the generation tasks and their checks ────────────────────────────────────


class _ConvertTask(BaseGenerationTask):
    """A CuratorKIT generation task that converts each prompt into ``target``.

    It builds the chat messages, cleans the reply, runs the checks, and records every rejected attempt by
    reason, length and hash. Subclasses set ``target``, ``system_prompt`` and ``script_check``.
    """

    target = ""
    system_prompt = ""

    def __init__(
        self,
        llm: BaseLLM,
        *,
        ratio_bounds: tuple[float, float] = DEFAULT_RATIO_BOUNDS,
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
        """The sample's language from its metadata; failing that "hi" if it has Devanagari, else "en"."""
        lang = sample.metadata.get("lang")
        if lang:
            return lang
        if contains_devanagari(sample.instruction):
            return "hi"
        return "en"

    def _build_messages(self, sample: DataSample) -> list[dict[str, str]]:
        """The system prompt, the few-shot examples as user/assistant turns, then this sample's request."""
        messages = [{"role": "system", "content": self.system_prompt}]
        for example_lang, example_source, example_output in FEWSHOT[self.target]:
            messages.append({"role": "user", "content": user_message(example_source, example_lang, self.target)})
            messages.append({"role": "assistant", "content": example_output})
        request = user_message(sample.instruction, self.source_lang(sample), self.target)
        messages.append({"role": "user", "content": request})
        return messages

    def script_check(self, text: str) -> str | None:
        """Reason to reject text written in the wrong script, or None."""
        raise NotImplementedError

    def check(self, source: str, text: str) -> str | None:
        """Reason to reject ``text`` as a conversion of ``source``, or None if it passes.

        The first failing check is the reason: "empty", the script check, "refusal", then "length_ratio:R"
        when R = len(text) / len(source) (both normalised) is outside ratio_bounds.
        """
        if not text:
            return "empty"
        script_problem = self.script_check(text)
        if script_problem:
            return script_problem
        if looks_like_refusal(text, source):
            return "refusal"
        ratio = len(normalize_text(text)) / max(len(normalize_text(source)), 1)
        ratio_min, ratio_max = self.ratio_bounds
        if not ratio_min <= ratio <= ratio_max:
            return f"length_ratio:{ratio:.2f}"
        return None

    def _parse_response(self, sample: DataSample, response) -> list[DataSample]:
        """The converted sample, or [] when a check rejects the reply (CuratorKIT then asks again)."""
        text = clean_output(getattr(response, "text", ""), sample.instruction)
        reason = self.check(sample.instruction, text)
        if reason:
            self._record_reject(sample.id, text, reason)
            return []
        self.last_reason.pop(sample.id, None)  # an earlier attempt may have failed
        return [self._converted_sample(sample, text)]

    def _record_reject(self, sample_id: str, text: str, reason: str) -> None:
        """Remember why this attempt failed, with its length and a hash; a text preview only if log_raw."""
        self.last_reason[sample_id] = reason
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:HASH_CHARS]
        self.attempts.setdefault(sample_id, []).append({"reason": reason, "chars": len(text), "sha256": text_hash})
        if self.log_raw:
            self.last_output[sample_id] = text[:PREVIEW_CHARS]

    def _converted_sample(self, sample: DataSample, text: str) -> DataSample:
        """The conversion as a new prompt-only sample, id ``<source id>-<target>``, with the source's metadata
        plus its language, source and converter."""
        metadata = dict(sample.metadata)
        metadata.update(
            lang=self.target,
            language=LANG_NAMES[self.target],
            source_id=sample.id,
            source_lang=self.source_lang(sample),
            converter_model=self.llm.model,
            converter_task=self.task_name,
        )
        return DataSample(
            id=f"{sample.id}-{self.target}",
            source_uri=sample.source_uri,
            instruction=text,
            task_type="prompt_only",
            metadata=metadata,
        )


class HinglishTask(_ConvertTask):
    """English or Hindi prompt -> romanised code-mixed Hinglish."""

    target = "hinglish"
    system_prompt = SYSTEM_HINGLISH

    def __init__(self, llm: BaseLLM, *, require_hinglish: bool = True, **task_options) -> None:
        super().__init__(llm, **task_options)
        self.require_hinglish = require_hinglish

    def script_check(self, text: str) -> str | None:
        """ "devanagari" if the text has any Devanagari character: Hinglish must be romanised."""
        if contains_devanagari(text):
            return "devanagari"
        return None

    def check(self, source: str, text: str) -> str | None:
        """The shared checks, then "not_hinglish" when required and the text has no romanised Hindi
        function words."""
        reason = super().check(source, text)
        if reason is not None:
            return reason
        if self.require_hinglish and not looks_hinglish(text):
            return "not_hinglish"
        return None


class HindiTranslationTask(_ConvertTask):
    """English prompt -> Hindi in Devanagari (for the XSTest safe prompts)."""

    target = "hi"
    system_prompt = SYSTEM_HINDI

    def script_check(self, text: str) -> str | None:
        """ "not_devanagari" if less than half of the letters are Devanagari."""
        if devanagari_fraction(text) < MIN_DEVANAGARI_FRACTION:
            return "not_devanagari"
        return None


TASKS = {"hinglish": HinglishTask, "hi": HindiTranslationTask}


# ── fake backend for --dry-run and tests ─────────────────────────────────────

# fmt: off
_ENGLISH_TO_HINGLISH = {
    "how": "kaise", "what": "kya", "why": "kyun", "where": "kahan", "when": "kab", "who": "kaun",
    "is": "hai", "are": "hain", "i": "main", "you": "aap", "me": "mujhe", "my": "mera", "your": "aapka",
    "and": "aur", "or": "ya", "not": "nahi", "in": "mein", "of": "ka", "for": "ke liye", "with": "ke saath",
    "to": "ko", "from": "se", "can": "sakta", "should": "chahiye", "tell": "batao", "make": "banao",
    "the": "", "a": "ek", "an": "ek", "this": "yeh", "that": "woh", "do": "karo", "please": "please",
}
_DEV_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
}
_DEV_VOWEL_SIGNS = {  # matras: a vowel written after a consonant
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॅ": "e", "ॉ": "o",
}
_DEV_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h", "ळ": "l",
}
_DEV_OTHER_SIGNS = {"ं": "n", "ँ": "n", "ः": "h", "़": "", "्": "", "।": ".", "॥": "."}
# fmt: on
_VIRAMA = "्"  # cancels a consonant's inherent "a"
_NUKTA = "़"  # the dot below a consonant
# The fake "Hindi" target: each Latin letter becomes some Devanagari letter (meaningless, but the right script).
_LATIN_TO_DEVANAGARI = dict(zip("abcdefghijklmnopqrstuvwxyz", "अबकदएफगहइजकलमनओपकरसटउववकयज"))
_USER_RE = re.compile(r"Convert this (\w+) text to (Hinglish|Hindi):\n\n(.*)\Z", re.S)  # see user_message


def romanize_devanagari(text: str) -> str:
    """Crude Devanagari -> Latin transliteration (good enough for a fake backend). A consonant gets its
    inherent "a" unless a vowel sign, virama or nukta follows it."""
    pieces: list[str] = []
    for i, char in enumerate(text):
        if char in _DEV_CONSONANTS:
            pieces.append(_DEV_CONSONANTS[char])
            next_char = text[i + 1 : i + 2]
            if next_char not in _DEV_VOWEL_SIGNS and next_char not in (_VIRAMA, _NUKTA):
                pieces.append("a")
        elif char in _DEV_VOWEL_SIGNS:
            pieces.append(_DEV_VOWEL_SIGNS[char])
        elif char in _DEV_VOWELS:
            pieces.append(_DEV_VOWELS[char])
        elif char in _DEV_OTHER_SIGNS:
            pieces.append(_DEV_OTHER_SIGNS[char])
        elif "०" <= char <= "९":
            pieces.append(str(ord(char) - ord("०")))  # Devanagari digit -> ASCII digit
        else:
            pieces.append(char)
    return "".join(pieces)


def _fake_hindi(text: str) -> str:
    """The text with each Latin letter swapped for a Devanagari one."""
    return "".join(_LATIN_TO_DEVANAGARI.get(char.lower(), char) for char in text)


def _fake_hinglish(text: str) -> str:
    """English text with common words swapped for Hindi ones, ending in " batao?" if that is not yet
    recognisably Hinglish."""
    words = []
    for word in text.split():
        swapped = _ENGLISH_TO_HINGLISH.get(word.strip(".,?!;:").lower(), word)
        if swapped:  # "the" maps to nothing
            words.append(swapped)
    converted = " ".join(words)
    if looks_hinglish(converted):
        return converted
    return converted.rstrip("?.!") + " batao?"


def _parse_user_message(content: str) -> tuple[str, str]:
    """(source text, target) from a user turn built by ``user_message``, target "hinglish" or "hi"; any other
    text is taken as a source to convert to Hinglish."""
    match = _USER_RE.search(content)
    if match is None:
        return content, "hinglish"
    target = match.group(2).lower()
    if target == "hindi":
        target = "hi"
    return match.group(3), target


class FakeLLM(BaseLLM):
    """Offline stand-in. By default: Hindi -> crude romanisation, English -> word-swap Hinglish, and any
    text -> "Hindi" as letter-by-letter Devanagari. ``script(source_text, target) -> reply`` overrides
    that, e.g. to inject refusals."""

    def __init__(self, script=None) -> None:
        super().__init__(model="fake/dry-run", temperature=0.0, max_tokens=512, max_retries=0)
        self.script = script
        self.calls = 0

    def _call(self, messages, **kwargs) -> LLMResponse:
        """Answer the last user turn, through ``script`` when one is set."""
        self.calls += 1
        source, target = _parse_user_message(messages[-1]["content"])
        return LLMResponse(text=self._reply(source, target), model=self.model)

    def _reply(self, source: str, target: str) -> str:
        """The scripted reply when a script is set, else the default fake conversion."""
        if self.script:
            return self.script(source, target)
        return self.default(source, target)

    @staticmethod
    def default(source: str, target: str) -> str:
        """The default fake conversion of ``source`` into ``target`` ("hi" or "hinglish")."""
        if target == "hi":
            return _fake_hindi(source)
        if contains_devanagari(source):
            return romanize_devanagari(source)
        return _fake_hinglish(source)


def make_llm(
    model: str | None, api_base: str | None, *, dry_run: bool, temperature: float = 0.3, max_tokens: int = 512
) -> BaseLLM:
    """The fake LLM for a dry run, else a LiteLLM backend for ``model`` with the key in $OPENAI_API_KEY
    (from the environment or .env)."""
    if dry_run:
        return FakeLLM()
    if not model:
        raise SystemExit("--model is required unless --dry-run (e.g. openai/CohereLabs/aya-expanse-8b)")
    from curatorkit.llm.litellm import LiteLLMBackend

    from pencil.config import load_env

    load_env()
    # Pass the key explicitly: given an api_base and no key, LiteLLMBackend sends the placeholder "nokey",
    # which the Hugging Face router rejects with HTTP 401.
    api_key = os.environ.get(API_KEY_ENV)
    return LiteLLMBackend(
        model=model, api_base=api_base, api_key=api_key, temperature=temperature, max_tokens=max_tokens
    )


# ── running a conversion over a set ──────────────────────────────────────────


def _row_to_sample(row: dict, *, path: str) -> DataSample:
    """A processed prompt row as a CuratorKIT prompt-only sample, keeping its id and metadata."""
    metadata = row.get("metadata")
    if not metadata:
        metadata = {}
    return DataSample(
        id=str(row["id"]),
        source_uri=f"file://{path}",
        instruction=str(row["prompt"]),
        task_type="prompt_only",
        metadata=dict(metadata),
    )


def run_split(task: _ConvertTask, inputs: list[Path], work_dir: Path):
    """One CuratorKIT Pipeline run: a JSONLReader per input file -> task -> PromptsMetaExporter.
    Writes manifest.json + lexsi_provenance.json (no rejected sidecar: it holds raw LLM text)."""
    from curatorkit.connectors.jsonl import JSONLReader
    from curatorkit.exporters.prompts_meta import PromptsMetaExporter
    from curatorkit.manifest import ProvenanceManifest, build_provenance, dataset_input, step_models
    from curatorkit.pipeline import Pipeline

    readers = []
    for path in inputs:
        to_sample = partial(_row_to_sample, path=str(path))
        readers.append(JSONLReader(path=path, format="prompt_only", preprocessing_fn=to_sample))
    steps = readers + [task, PromptsMetaExporter()]
    result = asyncio.run(Pipeline(steps, output_dir=work_dir).run_async())

    config_hash = task._config_hash()
    params = dict(step_models(steps))
    params["task_config_hash"] = config_hash
    params["ratio_bounds"] = list(task.ratio_bounds)
    provenance = build_provenance(
        method=f"pencil.{task.task_name}",
        inputs=[dataset_input(str(path)) for path in inputs],
        params=params,
    )
    ProvenanceManifest(result, config_hash, work_dir, provenance=provenance).write()
    return result


def _read_sources(inputs: list[Path]) -> list[dict]:
    """All rows of the source files, file after file; raises if any file is missing."""
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs {missing}; build them first")
    rows = []
    for path in inputs:
        rows.extend(read_jsonl(path))
    return rows


def _output_row(sample: DataSample, split: str) -> dict:
    """A converted sample as a processed prompt row, tagged with its split."""
    metadata = {**sample.metadata, "split": split}
    return {"id": sample.id, "prompt": sample.instruction, "metadata": metadata}


def _passed_rows(result, source_rows: list[dict], split: str) -> list[dict]:
    """The converted rows in the order of their sources, tagged with their split."""
    source_position = {row["id"]: position for position, row in enumerate(source_rows)}
    rows = [_output_row(sample, split) for sample in result.passed]
    return sorted(rows, key=lambda row: source_position[row["metadata"]["source_id"]])


def _log_fields(set_name: str, target: str, split: str) -> dict:
    """The fields every rejects-log entry starts with."""
    return {"set": set_name, "target": target, "split": split}


def _rejected_entries(result, task: _ConvertTask, where: dict) -> list[dict]:
    """One rejects-log entry per rejected sample: its id, reason and attempts (lengths and hashes only,
    plus a text preview when the task keeps them)."""
    entries = []
    for rejected in result.rejected:
        reason = task.last_reason.get(rejected.id)
        if not reason:  # rejected by CuratorKIT itself rather than by our checks
            reason = rejected.rejection_reason
        entry = {**where, "id": rejected.id, "reason": reason, "attempts": task.attempts.get(rejected.id, [])}
        if rejected.id in task.last_output:
            entry["output_preview"] = task.last_output[rejected.id]
        entries.append(entry)
    return entries


def _split_counts(n_source: int, split_rejects: list[dict]) -> dict:
    """A split's summary counts before the duplicate check: sources, rejects, and rejects per reason (the
    part before any ":", so "length_ratio:2.31" counts as "length_ratio")."""
    reasons = Counter(entry["reason"].split(":")[0] for entry in split_rejects)
    return {"n_source": n_source, "n_rejected": len(split_rejects), "rejected_by_reason": dict(reasons)}


def _drop_repeated_outputs(converted: dict[str, list[dict]]) -> dict[str, list[str]]:
    """An en and a hi prompt with the same meaning can convert to the same text. Keep each normalised
    output once, visiting test, then dev, then factor (``converted`` is changed in place); returns the
    dropped ids per split."""
    seen: set[str] = set()
    dropped: dict[str, list[str]] = {}
    for split in sorted(converted, key=SPLIT_KEEP_ORDER.index):
        kept = []
        dropped[split] = []
        for row in converted[split]:
            key = normalize_for_match(row["prompt"])
            if key in seen:
                dropped[split].append(row["id"])
            else:
                seen.add(key)
                kept.append(row)
        converted[split] = kept
    return dropped


def _duplicate_entry(where: dict, sample_id: str) -> dict:
    """The rejects-log entry of an output dropped as a repeat of one kept in another split."""
    return {**where, "id": sample_id, "reason": "duplicate_output", "attempts": []}


def _review_sample(rows: list[dict], frac: float, seed: str) -> list[dict]:
    """A seeded random ``frac`` of rows, at least one, for the human spot check (none when frac is 0)."""
    if not (rows and frac > 0):
        return []
    n_review = min(len(rows), max(1, math.ceil(frac * len(rows))))
    return random.Random(seed).sample(rows, n_review)


def _review_row(row: dict, set_name: str, source_prompts: dict[str, str], target: str) -> dict:
    """One spot-check CSV row: the source prompt beside its conversion, with blank verdict columns."""
    metadata = row["metadata"]
    return {
        "set": set_name,
        "split": metadata["split"],
        "id": row["id"],
        "source_id": metadata["source_id"],
        "source_lang": metadata["source_lang"],
        "source_prompt": source_prompts[metadata["source_id"]],
        target: row["prompt"],
        "ok (y/n)": "",
        "notes": "",
    }


def _write_review_csv(path: Path, rows: list[dict], target: str) -> None:
    """The spot-check CSV, with a UTF-8 BOM so Excel shows Devanagari correctly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["set", "split", "id", "source_id", "source_lang", "source_prompt", target, "ok (y/n)", "notes"]
    with open(path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _review_paths(review_dir: Path, target: str, set_name: str) -> tuple[Path, Path]:
    """The review CSV and rejects log, e.g. hinglish_check.csv and hinglish_rejects.jsonl; names for sets
    other than harmful end in _{set}."""
    suffix = "" if set_name == "harmful" else f"_{set_name}"
    return review_dir / f"{target}_check{suffix}.csv", review_dir / f"{target}_rejects{suffix}.jsonl"


def _task_options(
    target: str, ratio_bounds, log_raw: bool, concurrency: int, max_parse_retries: int, require_hinglish: bool
) -> dict:
    """Keyword arguments for the target's task class (require_hinglish applies to Hinglish only)."""
    options = {
        "ratio_bounds": tuple(ratio_bounds),
        "log_raw": log_raw,
        "concurrency": concurrency,
        "max_parse_retries": max_parse_retries,
    }
    if target == "hinglish":
        options["require_hinglish"] = require_hinglish
    return options


def _print_split_counts(target: str, set_name: str, split: str, counts: dict) -> None:
    """One progress line: kept/total, rejects (by reason) and duplicates dropped."""
    reasons = counts["rejected_by_reason"]
    reasons_text = str(reasons) if reasons else ""
    print(
        f"[{target}] {set_name}/{split}: {counts['n_out']}/{counts['n_source']} kept, {counts['n_rejected']} "
        f"rejected {reasons_text}, {counts['n_duplicate_output']} duplicate outputs dropped"
    )


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
    review_frac: float = DEFAULT_REVIEW_FRAC,
    seed: int = 0,
    ratio_bounds=DEFAULT_RATIO_BOUNDS,
    require_hinglish: bool = True,
    concurrency: int = 8,
    max_parse_retries: int = 2,
    log_raw: bool = False,
) -> dict:
    """Convert ``{set}/{source}/{split}.jsonl`` into ``{set}/{target}/{split}.jsonl`` under ``out_dir`` for
    every split. Also writes the review CSV, the rejects log and conversion.json; returns that summary."""
    if not splits:
        splits = PROMPT_SET_SPLITS[set_name]
    splits = list(splits)
    if not sources:
        sources = SET_SOURCES[set_name]
    sources = list(sources)
    log_raw = log_raw and set_name != "harmful"  # never keep raw text produced from harmful prompts
    task_options = _task_options(target, ratio_bounds, log_raw, concurrency, max_parse_retries, require_hinglish)
    out_set = out_dir / set_name / target
    summary = {"set": set_name, "target": target, "sources": sources, "model": llm.model, "splits": {}}
    converted: dict[str, list[dict]] = {}
    rejects: list[dict] = []
    source_prompts: dict[str, str] = {}

    # 1. Convert each split through CuratorKIT, with a fresh task; failed checks go to the rejects log.
    for split in splits:
        inputs = [data_dir / set_name / source / f"{split}.jsonl" for source in sources]
        source_rows = _read_sources(inputs)
        for row in source_rows:
            source_prompts[row["id"]] = row["prompt"]
        task = TASKS[target](llm, **task_options)
        result = run_split(task, inputs, out_set / "_curatorkit" / split)
        converted[split] = _passed_rows(result, source_rows, split)
        split_rejects = _rejected_entries(result, task, _log_fields(set_name, target, split))
        rejects += split_rejects
        summary["splits"][split] = _split_counts(len(source_rows), split_rejects)

    # 2. Keep each distinct output once across the splits.
    for split, dropped_ids in _drop_repeated_outputs(converted).items():
        where = _log_fields(set_name, target, split)
        rejects += [_duplicate_entry(where, sample_id) for sample_id in dropped_ids]
        summary["splits"][split]["n_duplicate_output"] = len(dropped_ids)
    if len(splits) < len(PROMPT_SET_SPLITS[set_name]):
        print(f"[{target}] note: duplicate outputs were only checked across {splits}")

    # 3. Write each split's conversions and draw its review sample.
    review_rows = []
    for split in splits:
        rows = converted[split]
        write_jsonl(out_set / f"{split}.jsonl", rows)
        review_sample = _review_sample(rows, review_frac, f"{seed}-{set_name}-{target}-{split}")
        review_rows += [_review_row(row, set_name, source_prompts, target) for row in review_sample]
        counts = summary["splits"][split]
        counts.update(n_out=len(rows), n_review=len(review_sample))
        _print_split_counts(target, set_name, split, counts)

    # 4. Write the review CSV, the rejects log and the summary.
    review_path, rejects_path = _review_paths(review_dir, target, set_name)
    _write_review_csv(review_path, review_rows, target)
    write_jsonl(rejects_path, rejects)
    summary.update(review_csv=str(review_path), rejects_log=str(rejects_path), created=timestamp())
    write_manifest(out_set / "conversion.json", summary)
    print(f"[{target}] review sample -> {review_path}; rejects -> {rejects_path}")
    return summary


def conversion_problems(summary: dict) -> list[str]:
    """One message per split that converted nothing, or lost more than MAX_REJECTED_FRACTION of its rows
    to rejects."""
    problems = []
    for split, counts in summary["splits"].items():
        where = f"{summary['set']}/{summary['target']}/{split}"
        if counts["n_out"] == 0:
            problems.append(f"{where}: no rows converted")
        elif counts["n_rejected"] > MAX_REJECTED_FRACTION * counts["n_source"]:
            problems.append(f"{where}: {counts['n_rejected']} of {counts['n_source']} rows rejected")
    return problems


def fail_on_problems(summary: dict) -> None:
    """Exit non-zero if the conversion looks broken, so a job chain stops instead of using its output."""
    problems = conversion_problems(summary)
    if problems:
        details = "\n  ".join(problems)
        raise SystemExit(f"conversion failed:\n  {details}\nreasons: {summary['rejects_log']}")


# ── command line ─────────────────────────────────────────────────────────────


def add_llm_args(parser: argparse.ArgumentParser) -> None:
    """The LLM and check options, shared with 01_build_xstest.py."""
    group = parser.add_argument_group("LLM")
    group.add_argument(
        "--model",
        default=None,
        help="LiteLLM model, e.g. openai/CohereLabs/aya-expanse-8b or cohere_chat/c4ai-aya-expanse-32b",
    )
    group.add_argument("--api-base", default=None, help="e.g. http://localhost:8000/v1 for a local vLLM server")
    group.add_argument("--temperature", type=float, default=0.3, help="> 0 so a retry can differ")
    group.add_argument("--max-tokens", type=int, default=512)
    group.add_argument("--concurrency", type=int, default=8)
    group.add_argument("--max-parse-retries", type=int, default=2, help="extra attempts after a rejected output")
    group.add_argument("--ratio-min", type=float, default=DEFAULT_RATIO_BOUNDS[0])
    group.add_argument("--ratio-max", type=float, default=DEFAULT_RATIO_BOUNDS[1])
    group.add_argument(
        "--no-require-hinglish",
        action="store_true",
        help="do not reject outputs without romanised Hindi function words",
    )
    group.add_argument("--dry-run", action="store_true", help="fake offline LLM; writes under outputs/dry_run/")
    group.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    group.add_argument("--review-frac", type=float, default=DEFAULT_REVIEW_FRAC)


def parse_args(argv=None):
    """The command-line options (see the module docstring)."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__
    )
    parser.add_argument("--set", dest="set_name", choices=sorted(PROMPT_SET_SPLITS), default="harmful")
    parser.add_argument("--splits", nargs="+", default=None, help="default: every split of the set")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["en", "hi"],
        default=None,
        help="source languages (default: en hi for harmful, en for xstest)",
    )
    parser.add_argument("--data-dir", default=None, help="processed data root to read (default: data/processed)")
    parser.add_argument(
        "--out-dir", default=None, help="root to write (default: --data-dir; outputs/dry_run with --dry-run)"
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for the review sample")
    parser.add_argument(
        "--log-raw-rejects",
        action="store_true",
        help="keep a 300-char preview of rejected outputs (xstest only; ignored for harmful)",
    )
    add_llm_args(parser)
    return parser.parse_args(argv)


def _output_root(args: argparse.Namespace, data_dir: Path) -> Path:
    """--out-dir if given; else outputs/dry_run/processed for a dry run (never real data); else the data dir."""
    if args.out_dir:
        return Path(args.out_dir)
    if args.dry_run:
        return DRY_RUN_DIR
    return data_dir


def main(argv=None) -> int:
    args = parse_args(argv)
    data_dir = resolve_data_dir(args.data_dir)
    out_dir = _output_root(args, data_dir)
    if args.log_raw_rejects and args.set_name == "harmful":
        print("[hinglish] --log-raw-rejects is ignored for the harmful set")
    llm = make_llm(
        args.model, args.api_base, dry_run=args.dry_run, temperature=args.temperature, max_tokens=args.max_tokens
    )
    summary = run_conversion(
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
    fail_on_problems(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
