"""Batched greedy generation through the chat template (one user turn plus the generation prompt).

Two back ends give the same strings for the same weights (tests/test_evaluate.py checks this):

- a path, hub id or PEFT adapter folder is loaded through AuditKIT (`AutoModel.resolve("hf:...")`) and sent the
  `Request` that AuditKIT's GenerationAdapter would build. We call the model directly rather than `ak.generate`,
  so a failed batch raises instead of becoming empty completions, and AuditKIT's run cache is never involved.
  The model is unloaded before `generate` returns;
- an in-memory model goes through `model.generate` on left-padded `pencil.chat.template_ids`.

`generate` writes nothing. `generate_cached` keeps completions in outputs/generations/ (gitignored); those
include harmful completions, so never commit, upload or share them (CLAUDE.md rule 7).
"""

from __future__ import annotations

import gc
import hashlib
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from . import chat
from .config import REPO_ROOT
from .datasets import read_jsonl, write_jsonl

GENERATIONS_DIR = REPO_ROOT / "outputs" / "generations"

# A checkpoint folder's tokenizer must render this exactly like the pinned tokenizer (Latin and Devanagari).
_TEMPLATE_PROBE = "Template check: नमस्ते, kaise ho?"
# AuditKIT's GenerationAdapter fixes this seed in every request; we do the same so requests match.
_AUDITKIT_SEED = 0
# Runs of characters that may not appear in a cache directory name.
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def sha256(text: str) -> str:
    """Hex SHA-256 of the UTF-8 text; results store this instead of a completion."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def default_dtype() -> torch.dtype:
    """bfloat16 on a GPU, float32 on the CPU."""
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def free_gpu() -> None:
    """Collect garbage and empty the CUDA cache, so the memory of models that went out of scope is released."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pad_id(tokenizer) -> int:
    """The tokenizer's pad id, or its EOS id when it has none."""
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    return tokenizer.eos_token_id


def prompt_ids(tokenizer, prompt: str) -> list[int]:
    """The ids the model continues: the chat template around one user turn, plus the generation prompt."""
    messages = [{"role": "user", "content": prompt}]
    return chat.template_ids(tokenizer, messages, add_generation_prompt=True)


def longest_first_batches(lengths: Sequence[int], batch_size: int) -> Iterator[list[int]]:
    """Batches of indices, longest first, so each batch holds similar lengths (less padding)."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def generate(
    model_or_path: Any,
    tokenizer,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 256,
    batch_size: int = 16,
    temperature: float = 0.0,
    load_kwargs: dict | None = None,
) -> list[str]:
    """Complete each prompt (greedy when temperature == 0), in prompt order.

    `load_kwargs` go to `AutoModel.resolve` for a path (e.g. device="cpu", dtype=torch.float32).
    """
    prompts = list(prompts)
    if not prompts:
        return []
    if isinstance(model_or_path, (str, os.PathLike)):
        return _generate_auditkit(
            str(model_or_path),
            tokenizer,
            prompts,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            temperature=temperature,
            load_kwargs=load_kwargs,
        )
    return _generate_hf(
        model_or_path,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        temperature=temperature,
    )


# ------------------------------------------------------------------------------ in-memory model


def _hf_generate_kwargs(model, tokenizer, max_new_tokens: int, temperature: float) -> dict[str, Any]:
    """Keyword arguments for `model.generate`: greedy when temperature is 0, else sampling at that temperature."""
    eos_token_id = model.generation_config.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": pad_id(tokenizer),
        "eos_token_id": eos_token_id,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature
    return kwargs


@torch.no_grad()
def _generate_hf(
    model, tokenizer, prompts: list[str], *, max_new_tokens: int, batch_size: int, temperature: float
) -> list[str]:
    """Completions from `model.generate` on left-padded batches of templated prompts, in prompt order."""
    generate_kwargs = _hf_generate_kwargs(model, tokenizer, max_new_tokens, temperature)
    device = next(model.parameters()).device
    encoded_prompts = [prompt_ids(tokenizer, prompt) for prompt in prompts]
    prompt_lengths = [len(ids) for ids in encoded_prompts]
    completions = [""] * len(prompts)

    was_training = model.training
    model.eval()
    try:
        for batch_indices in longest_first_batches(prompt_lengths, batch_size):
            encoded = [chat.Encoded(encoded_prompts[i], 0, prompt_lengths[i]) for i in batch_indices]
            batch = chat.collate(encoded, pad_id(tokenizer), side="left")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
            new_token_ids = output_ids[:, input_ids.shape[1] :]
            texts = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
            for i, text in zip(batch_indices, texts):
                completions[i] = text
    finally:
        model.train(was_training)
    return completions


# ------------------------------------------------------------------------------ AuditKIT model


def ensure_pad_token(model: Any) -> None:
    """Give an AuditKIT HFGenModel's tokenizer a pad token (EOS) if it has none, as a batched pipeline needs
    one (Llama Guard 3 has none). A no-op for any other model; safe to repeat and after `.unload()`."""
    load_pipeline = getattr(model, "_ensure_pipeline", None)
    if load_pipeline is None:
        return
    load_pipeline()
    pipeline_tokenizer = getattr(model._pipeline, "tokenizer", None)
    if pipeline_tokenizer is not None and pipeline_tokenizer.pad_token_id is None:
        pipeline_tokenizer.pad_token = pipeline_tokenizer.eos_token


def _first_reply_text(result: Any) -> str:
    """The text of an AuditKIT result's first completion, or "" when it has none."""
    if not result.completions:
        return ""
    text = result.completions[0].text
    if not text:
        return ""
    return text


def generate_requests(model: Any, requests: list, lengths: Sequence[int], batch_size: int) -> list[str]:
    """Send AuditKIT requests in longest-first batches (by `lengths`); the first reply text of each, in order."""
    texts = [""] * len(requests)
    for batch_indices in longest_first_batches(lengths, batch_size):
        batch_requests = [requests[i] for i in batch_indices]
        results = model.generate(batch_requests)
        for i, result in zip(batch_indices, results):
            texts[i] = _first_reply_text(result)
    return texts


def _check_same_template(model: Any, tokenizer) -> None:
    """A checkpoint folder's tokenizer must render prompts exactly like the pinned one, so a PEFT folder
    cannot bring a different chat template."""
    pipeline = getattr(model, "_pipeline", None)
    checkpoint_tokenizer = getattr(pipeline, "tokenizer", None)
    if checkpoint_tokenizer is None:
        same_template = False
    else:
        same_template = prompt_ids(checkpoint_tokenizer, _TEMPLATE_PROBE) == prompt_ids(tokenizer, _TEMPLATE_PROBE)
    if not same_template:
        model_name = getattr(model, "_model_name", model)
        raise ValueError(
            f"{model_name!r}: its tokenizer / chat template differs from "
            "the pinned tokenizer; copy the pinned tokenizer files into the checkpoint folder"
        )


def _generate_auditkit(
    spec: str,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int,
    batch_size: int,
    temperature: float,
    load_kwargs: dict | None,
) -> list[str]:
    """Load the checkpoint through AuditKIT, complete the prompts, and unload it again (even on error)."""
    from auditkit.model import AutoModel, Request

    if not spec.startswith("hf:"):
        spec = "hf:" + spec
    resolve_kwargs = {"dtype": default_dtype()}
    if load_kwargs:
        resolve_kwargs.update(load_kwargs)
    model = AutoModel.resolve(spec, batch_size=batch_size, **resolve_kwargs)

    # The same generation parameters AuditKIT's GenerationAdapter puts in each request.
    request_params = {"temperature": temperature, "max_tokens": max_new_tokens, "seed": _AUDITKIT_SEED}
    try:
        ensure_pad_token(model)
        _check_same_template(model, tokenizer)
        requests = [Request(prompt=prompt, params=dict(request_params)) for prompt in prompts]
        lengths = [len(prompt_ids(tokenizer, prompt)) for prompt in prompts]
        return generate_requests(model, requests, lengths, batch_size)
    finally:
        model.unload()
        free_gpu()


# ------------------------------------------------------------------------------------------ cache


def checkpoint_name(checkpoint: str) -> str:
    """One directory name per checkpoint: a path inside the repo is taken relative to it, then every run of
    characters outside [A-Za-z0-9._-] becomes "__" (hub ids and paths alike)."""
    name = str(checkpoint).removeprefix("hf:")
    path = Path(name)
    if path.exists():
        path = path.resolve()
        if path.is_relative_to(REPO_ROOT):
            path = path.relative_to(REPO_ROOT)
        name = str(path)
    safe_name = _UNSAFE_NAME_CHARS.sub("__", name).strip("_.")
    if not safe_name:
        return "model"
    return safe_name


def generations_path(checkpoint: str, split: str, root: str | Path | None = None) -> Path:
    """The cache file {root}/{checkpoint name}/{split}.jsonl; root defaults to outputs/generations."""
    if not root:
        root = GENERATIONS_DIR
    return Path(root) / checkpoint_name(checkpoint) / f"{split}.jsonl"


def load_generations(path: str | Path) -> dict[str, dict]:
    """{id: row} from one cache file ({} if it does not exist)."""
    if not Path(path).exists():
        return {}
    return {row["id"]: row for row in read_jsonl(path)}


def _is_reusable(cached_row: dict | None, row: Mapping[str, Any], settings: dict) -> bool:
    """True if a cached row was generated from the same prompt text with the same generation settings."""
    if not cached_row:
        return False
    same_prompt = cached_row.get("prompt_sha256") == sha256(row["prompt"])
    same_settings = cached_row.get("gen") == settings
    return same_prompt and same_settings


def _rows_to_generate(
    splits: Mapping[str, Sequence[Mapping[str, Any]]], caches: dict[str, dict], settings: dict
) -> list[tuple[str, Mapping[str, Any]]]:
    """The (split, row) pairs with no reusable cached completion; raises on a duplicate id within a split."""
    todo = []
    for split, rows in splits.items():
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"duplicate ids in split {split!r}")
        for row in rows:
            if not _is_reusable(caches[split].get(row["id"]), row, settings):
                todo.append((split, row))
    return todo


def _cache_row(row: Mapping[str, Any], completion: str, settings: dict) -> dict:
    """One cache line: the prompt, the completion, their hashes and the generation settings."""
    return {
        "id": row["id"],
        "prompt": row["prompt"],
        "prompt_sha256": sha256(row["prompt"]),
        "completion": completion,
        "completion_sha256": sha256(completion),
        "gen": settings,
    }


def generate_cached(
    model_or_path: Any,
    tokenizer,
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    checkpoint: str,
    max_new_tokens: int = 256,
    batch_size: int = 16,
    temperature: float = 0.0,
    load_kwargs: dict | None = None,
    root: str | Path | None = None,
) -> dict[str, dict[str, str]]:
    """Generate for {split: [{"id", "prompt"}, ...]}, reusing outputs/generations/{checkpoint}/{split}.jsonl.

    A cached row is reused only if its prompt hash and generation settings match; everything else is
    generated in one `generate` call (one model load). Returns {split: {id: completion}}.
    """
    settings = {"max_new_tokens": max_new_tokens, "temperature": temperature}
    caches = {split: load_generations(generations_path(checkpoint, split, root)) for split in splits}
    todo = _rows_to_generate(splits, caches, settings)

    if todo:
        texts = generate(
            model_or_path,
            tokenizer,
            [row["prompt"] for _, row in todo],
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            temperature=temperature,
            load_kwargs=load_kwargs,
        )
        for (split, row), text in zip(todo, texts):
            caches[split][row["id"]] = _cache_row(row, text, settings)
        changed_splits = dict.fromkeys(split for split, _ in todo)  # each split once, in first-seen order
        for split in changed_splits:
            write_jsonl(generations_path(checkpoint, split, root), caches[split].values())

    completions: dict[str, dict[str, str]] = {}
    for split, rows in splits.items():
        completions[split] = {row["id"]: caches[split][row["id"]]["completion"] for row in rows}
    return completions
