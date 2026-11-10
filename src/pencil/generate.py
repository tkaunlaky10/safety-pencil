"""Batched greedy generation through the chat template, for an in-memory HF model or a path / hub id.

Two back ends, one prompt format (a single user turn, `add_generation_prompt=True`):

- a path, hub id or PEFT adapter folder goes through AuditKIT: `AutoModel.resolve("hf:...", batch_size=...)`
  and the same `Request` its GenerationAdapter builds. We call the model directly instead of `ak.generate`
  so a failed batch raises rather than turning into empty completions, and AuditKIT's run cache is never
  involved. The AuditKIT tokenizer must render prompts exactly like ours (checked before generating).
- an in-memory model goes through `model.generate` on `pencil.chat.template_ids`, left-padded.

Both decode only the new tokens with `skip_special_tokens=True`, so the same weights give the same strings
(tests/test_evaluate.py checks this on the tiny model).

`generate` never writes anything. `generate_cached` is the opt-in cache in outputs/generations/ (gitignored).
That cache holds raw completions, harmful ones included: never commit, upload or share it (CLAUDE.md rule 7).
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from . import chat
from .config import REPO_ROOT

GENERATIONS_DIR = REPO_ROOT / "outputs" / "generations"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def default_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prompt_ids(tok, prompt: str) -> list[int]:
    """The ids the model continues: the chat template around one user turn, plus the generation prompt."""
    return chat.template_ids(tok, [{"role": "user", "content": prompt}], add_generation_prompt=True)


def generate(model_or_path: Any, tokenizer, prompts: Sequence[str], *, max_new_tokens: int = 256,
             batch_size: int = 16, temperature: float = 0.0, load_kwargs: dict | None = None) -> list[str]:
    """Complete each prompt (greedy when temperature == 0). Returns completions in prompt order.

    `load_kwargs` go to `AutoModel.resolve` for a path (e.g. device="cpu", dtype=torch.float32).
    """
    prompts = list(prompts)
    if not prompts:
        return []
    if isinstance(model_or_path, (str, os.PathLike)):
        return _generate_auditkit(str(model_or_path), tokenizer, prompts, max_new_tokens=max_new_tokens,
                                  batch_size=batch_size, temperature=temperature, load_kwargs=load_kwargs)
    return _generate_hf(model_or_path, tokenizer, prompts, max_new_tokens=max_new_tokens,
                        batch_size=batch_size, temperature=temperature)


def _longest_first(lengths: Sequence[int]) -> list[int]:
    """Indices sorted by decreasing length, so each batch holds similar lengths (less padding)."""
    return sorted(range(len(lengths)), key=lambda i: -lengths[i])


@torch.no_grad()
def _generate_hf(model, tok, prompts: list[str], *, max_new_tokens: int, batch_size: int,
                 temperature: float) -> list[str]:
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos = model.generation_config.eos_token_id
    kw: dict[str, Any] = {"max_new_tokens": max_new_tokens, "pad_token_id": pad,
                          "eos_token_id": eos if eos is not None else tok.eos_token_id,
                          "do_sample": temperature > 0}
    if temperature > 0:
        kw["temperature"] = temperature
    device = next(model.parameters()).device
    enc = [prompt_ids(tok, p) for p in prompts]
    out = [""] * len(prompts)
    was_training = model.training
    model.eval()
    try:
        order = _longest_first([len(e) for e in enc])
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            batch = chat.collate([chat.Encoded(enc[i], 0, len(enc[i])) for i in idx], pad, side="left")
            ids = batch["input_ids"].to(device)
            gen = model.generate(input_ids=ids, attention_mask=batch["attention_mask"].to(device), **kw)
            for i, text in zip(idx, tok.batch_decode(gen[:, ids.shape[1]:], skip_special_tokens=True)):
                out[i] = text
    finally:
        model.train(was_training)
    return out


def ensure_pad_token(model: Any) -> None:
    """A batched HF pipeline needs a pad token (Llama Guard 3's tokenizer has none): pad with EOS.

    Only for AuditKIT's HFGenModel; a no-op for anything else. Idempotent, and safe after `.unload()`.
    """
    load = getattr(model, "_ensure_pipeline", None)
    if load is None:
        return
    load()
    tok = getattr(model._pipeline, "tokenizer", None)
    if tok is not None and tok.pad_token_id is None:
        tok.pad_token = tok.eos_token


def _check_same_template(model: Any, tok) -> None:
    """The checkpoint folder's tokenizer must render prompts exactly like the pinned one (CLAUDE.md:
    a PEFT folder must not bring a different template)."""
    theirs = getattr(getattr(model, "_pipeline", None), "tokenizer", None)
    probe = "Template check: नमस्ते, kaise ho?"
    if theirs is None or prompt_ids(theirs, probe) != prompt_ids(tok, probe):
        raise ValueError(f"{getattr(model, '_model_name', model)!r}: its tokenizer / chat template differs from "
                         "the pinned tokenizer; copy the pinned tokenizer files into the checkpoint folder")


def _generate_auditkit(spec: str, tok, prompts: list[str], *, max_new_tokens: int, batch_size: int,
                       temperature: float, load_kwargs: dict | None) -> list[str]:
    from auditkit.model import AutoModel, Request

    spec = spec if spec.startswith("hf:") else "hf:" + spec
    model = AutoModel.resolve(spec, batch_size=batch_size, **{"dtype": default_dtype(), **(load_kwargs or {})})
    params = {"temperature": temperature, "max_tokens": max_new_tokens, "seed": 0}   # as GenerationAdapter
    out = [""] * len(prompts)
    try:
        ensure_pad_token(model)
        _check_same_template(model, tok)
        order = _longest_first([len(prompt_ids(tok, p)) for p in prompts])
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            results = model.generate([Request(prompt=prompts[i], params=dict(params)) for i in idx])
            for i, r in zip(idx, results):
                out[i] = r.completions[0].text if r.completions else ""
    finally:
        model.unload()
        free_gpu()
    return out


# ------------------------------------------------------------------------------------------ cache

def checkpoint_name(checkpoint: str) -> str:
    """One directory name per checkpoint: a path inside the repo is taken relative to it, then
    everything outside [A-Za-z0-9._-] becomes "__" (hub ids and paths alike)."""
    s = str(checkpoint).removeprefix("hf:")
    p = Path(s)
    if p.exists():
        try:
            s = str(p.resolve().relative_to(REPO_ROOT))
        except ValueError:
            s = str(p.resolve())
    return re.sub(r"[^A-Za-z0-9._-]+", "__", s).strip("_.") or "model"


def generations_path(checkpoint: str, split: str, root: str | Path | None = None) -> Path:
    return Path(root or GENERATIONS_DIR) / checkpoint_name(checkpoint) / f"{split}.jsonl"


def load_generations(path: str | Path) -> dict[str, dict]:
    """{id: row} from one cache file ({} if it does not exist)."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {row["id"]: row for row in map(json.loads, filter(str.strip, f))}


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def generate_cached(model_or_path: Any, tokenizer, splits: Mapping[str, Sequence[Mapping[str, Any]]], *,
                    checkpoint: str, max_new_tokens: int = 256, batch_size: int = 16, temperature: float = 0.0,
                    load_kwargs: dict | None = None, root: str | Path | None = None) -> dict[str, dict[str, str]]:
    """Generate for {split: [{"id", "prompt"}, ...]}, reusing outputs/generations/{checkpoint}/{split}.jsonl.

    A cached row is reused only if its prompt hash and generation settings match; everything missing is
    generated in one `generate` call (one model load). Returns {split: {id: completion}}.
    """
    settings = {"max_new_tokens": max_new_tokens, "temperature": temperature}
    caches = {s: load_generations(generations_path(checkpoint, s, root)) for s in splits}
    todo: list[tuple[str, Mapping[str, Any]]] = []
    for s, rows in splits.items():
        ids = [r["id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate ids in split {s!r}")
        for r in rows:
            hit = caches[s].get(r["id"])
            if not (hit and hit.get("prompt_sha256") == sha256(r["prompt"]) and hit.get("gen") == settings):
                todo.append((s, r))
    if todo:
        texts = generate(model_or_path, tokenizer, [r["prompt"] for _, r in todo], max_new_tokens=max_new_tokens,
                         batch_size=batch_size, temperature=temperature, load_kwargs=load_kwargs)
        for (s, r), text in zip(todo, texts):
            caches[s][r["id"]] = {"id": r["id"], "prompt": r["prompt"], "prompt_sha256": sha256(r["prompt"]),
                                  "completion": text, "completion_sha256": sha256(text), "gen": settings}
        for s in dict.fromkeys(s for s, _ in todo):
            _write_jsonl(generations_path(checkpoint, s, root), caches[s].values())
    return {s: {r["id"]: caches[s][r["id"]]["completion"] for r in rows} for s, rows in splits.items()}
