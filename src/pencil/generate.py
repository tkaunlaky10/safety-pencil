"""Batched greedy generation through the chat template (one user turn plus the generation prompt).

Two back ends give the same strings for the same weights (tests/test_evaluate.py checks this):

- a path, hub id or PEFT adapter folder goes through AuditKIT's model (`AutoModel.resolve("hf:...")`) with the
  `Request` its GenerationAdapter builds. We call the model directly rather than `ak.generate`, so a failed
  batch raises instead of becoming empty completions and AuditKIT's run cache is never involved;
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


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def default_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pad_id(tok) -> int:
    """The tokenizer's pad id, or its EOS id when it has none."""
    return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def prompt_ids(tok, prompt: str) -> list[int]:
    """The ids the model continues: the chat template around one user turn, plus the generation prompt."""
    return chat.template_ids(tok, [{"role": "user", "content": prompt}], add_generation_prompt=True)


def longest_first_batches(lengths: Sequence[int], batch_size: int) -> Iterator[list[int]]:
    """Batches of indices, longest first, so each batch holds similar lengths (less padding)."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    for s in range(0, len(order), batch_size):
        yield order[s : s + batch_size]


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
        model_or_path, tokenizer, prompts, max_new_tokens=max_new_tokens, batch_size=batch_size, temperature=temperature
    )


@torch.no_grad()
def _generate_hf(
    model, tok, prompts: list[str], *, max_new_tokens: int, batch_size: int, temperature: float
) -> list[str]:
    eos = model.generation_config.eos_token_id
    kw: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": pad_id(tok),
        "eos_token_id": eos if eos is not None else tok.eos_token_id,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        kw["temperature"] = temperature
    device = next(model.parameters()).device
    enc = [prompt_ids(tok, p) for p in prompts]
    out = [""] * len(prompts)
    was_training = model.training
    model.eval()
    try:
        for idx in longest_first_batches([len(e) for e in enc], batch_size):
            batch = chat.collate([chat.Encoded(enc[i], 0, len(enc[i])) for i in idx], pad_id(tok), side="left")
            ids = batch["input_ids"].to(device)
            gen = model.generate(input_ids=ids, attention_mask=batch["attention_mask"].to(device), **kw)
            for i, text in zip(idx, tok.batch_decode(gen[:, ids.shape[1] :], skip_special_tokens=True)):
                out[i] = text
    finally:
        model.train(was_training)
    return out


def ensure_pad_token(model: Any) -> None:
    """Give an AuditKIT HFGenModel's tokenizer a pad token (EOS) if it has none, as a batched pipeline needs
    one (Llama Guard 3 has none). A no-op for any other model; safe to repeat and after `.unload()`."""
    load = getattr(model, "_ensure_pipeline", None)
    if load is None:
        return
    load()
    tok = getattr(model._pipeline, "tokenizer", None)
    if tok is not None and tok.pad_token_id is None:
        tok.pad_token = tok.eos_token


def generate_requests(model: Any, requests: list, lengths: Sequence[int], batch_size: int) -> list[str]:
    """Send AuditKIT requests in longest-first batches (by `lengths`); the first reply text of each, in order."""
    texts = [""] * len(requests)
    for idx in longest_first_batches(lengths, batch_size):
        for i, result in zip(idx, model.generate([requests[i] for i in idx])):
            texts[i] = (result.completions[0].text or "") if result.completions else ""
    return texts


def _check_same_template(model: Any, tok) -> None:
    """A checkpoint folder's tokenizer must render prompts exactly like the pinned one, so a PEFT folder
    cannot bring a different chat template."""
    theirs = getattr(getattr(model, "_pipeline", None), "tokenizer", None)
    probe = "Template check: नमस्ते, kaise ho?"
    if theirs is None or prompt_ids(theirs, probe) != prompt_ids(tok, probe):
        raise ValueError(
            f"{getattr(model, '_model_name', model)!r}: its tokenizer / chat template differs from "
            "the pinned tokenizer; copy the pinned tokenizer files into the checkpoint folder"
        )


def _generate_auditkit(
    spec: str,
    tok,
    prompts: list[str],
    *,
    max_new_tokens: int,
    batch_size: int,
    temperature: float,
    load_kwargs: dict | None,
) -> list[str]:
    from auditkit.model import AutoModel, Request

    spec = spec if spec.startswith("hf:") else "hf:" + spec
    model = AutoModel.resolve(spec, batch_size=batch_size, **{"dtype": default_dtype(), **(load_kwargs or {})})
    params = {"temperature": temperature, "max_tokens": max_new_tokens, "seed": 0}  # as GenerationAdapter
    try:
        ensure_pad_token(model)
        _check_same_template(model, tok)
        requests = [Request(prompt=p, params=dict(params)) for p in prompts]
        return generate_requests(model, requests, [len(prompt_ids(tok, p)) for p in prompts], batch_size)
    finally:
        model.unload()
        free_gpu()


# ------------------------------------------------------------------------------------------ cache


def checkpoint_name(checkpoint: str) -> str:
    """One directory name per checkpoint: a path inside the repo is taken relative to it, then every run of
    characters outside [A-Za-z0-9._-] becomes "__" (hub ids and paths alike)."""
    s = str(checkpoint).removeprefix("hf:")
    p = Path(s)
    if p.exists():
        p = p.resolve()
        s = str(p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p)
    return re.sub(r"[^A-Za-z0-9._-]+", "__", s).strip("_.") or "model"


def generations_path(checkpoint: str, split: str, root: str | Path | None = None) -> Path:
    return Path(root or GENERATIONS_DIR) / checkpoint_name(checkpoint) / f"{split}.jsonl"


def load_generations(path: str | Path) -> dict[str, dict]:
    """{id: row} from one cache file ({} if it does not exist)."""
    return {row["id"]: row for row in read_jsonl(path)} if Path(path).exists() else {}


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
    caches = {s: load_generations(generations_path(checkpoint, s, root)) for s in splits}
    todo: list[tuple[str, Mapping[str, Any]]] = []
    for s, rows in splits.items():
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"duplicate ids in split {s!r}")
        for r in rows:
            hit = caches[s].get(r["id"])
            if not (hit and hit.get("prompt_sha256") == sha256(r["prompt"]) and hit.get("gen") == settings):
                todo.append((s, r))
    if todo:
        texts = generate(
            model_or_path,
            tokenizer,
            [r["prompt"] for _, r in todo],
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            temperature=temperature,
            load_kwargs=load_kwargs,
        )
        for (s, r), text in zip(todo, texts):
            caches[s][r["id"]] = {
                "id": r["id"],
                "prompt": r["prompt"],
                "prompt_sha256": sha256(r["prompt"]),
                "completion": text,
                "completion_sha256": sha256(text),
                "gen": settings,
            }
        for s in dict.fromkeys(s for s, _ in todo):
            write_jsonl(generations_path(checkpoint, s, root), caches[s].values())
    return {s: {r["id"]: caches[s][r["id"]]["completion"] for r in rows} for s, rows in splits.items()}
