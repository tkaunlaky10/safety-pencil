"""Per-language evaluation of one checkpoint (Stage 8).

Metrics, per language: refusal and harmful-compliance (Llama Guard "unsafe") rates on harmful prompts, and
the over-refusal rate on XSTest-derived benign prompts, each with a Wilson 95% interval; Global-MMLU
log-likelihood accuracy; mean per-token KL(p_reference || p_checkpoint) on the benign prompts (only with a
reference). Plus the per-token NLL of held-out task answers.

Order, so that a checkpoint given as a path never shares the GPU with a judge:
  1. generate on harmful + benign prompts (cached in outputs/generations/ when cache=True);
  2. teacher-forced metrics, with the checkpoint (and reference) loaded once and freed after;
  3. Llama Guard 3 on the harmful responses, then unloaded;
  4. the refusal judge on harmful + benign responses, then unloaded.
An in-memory model stays loaded throughout (its owner frees it), so pass a path for GPU runs.

Results hold ids, labels and sha256 hashes of completions, never completion text (CLAUDE.md rule 7).
"""
from __future__ import annotations

import math
import os
from typing import Any, Iterator, Mapping, Sequence

import torch

from . import chat
from .generate import default_dtype, free_gpu, generate, generate_cached, prompt_ids, sha256
from .judge import (DEFAULT_HARM_JUDGE, DEFAULT_REFUSAL_JUDGE, REFUSAL_RUBRIC, harm_judge, judge_name,
                    read_rubric, refusal_judge)
from .results import write_result

TIER_SPLIT = {"probe": "dev", "full": "test"}       # CLAUDE.md rule 5
LETTERS = "ABCDEFGHIJ"


# ------------------------------------------------------------------------------- data selection

def select_tier(data: Mapping[str, Any], tier: str, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The items one tier evaluates. probe: the first `evaluation.probe_per_lang` items of every list
    (MMLU also capped at `mmlu_per_lang`); full: everything given (MMLU capped at `mmlu_per_lang`)."""
    if tier not in TIER_SPLIT:
        raise ValueError(f"tier must be one of {sorted(TIER_SPLIT)}, got {tier!r}")
    ev = cfg.get("evaluation", {})
    cap = ev.get("probe_per_lang") if tier == "probe" else None
    caps = [c for c in (cap, ev.get("mmlu_per_lang")) if c]
    mmlu_cap = min(caps) if caps else None

    def take(rows: Sequence[Mapping], n: int | None, what: str) -> list:
        rows = list(rows)[:n] if n else list(rows)
        ids = [r.get("id") for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate or missing ids in {what}")
        return rows

    return {
        "harmful": {l: take(r, cap, f"harmful/{l}") for l, r in data.get("harmful", {}).items()},
        "xstest": {l: take(r, cap, f"xstest/{l}") for l, r in data.get("xstest", {}).items()},
        "task": take(data.get("task", []), cap, "task"),
        "mmlu": {l: take(r, mmlu_cap, f"mmlu/{l}") for l, r in data.get("mmlu", {}).items()},
    }


# ---------------------------------------------------------------------------- teacher forcing

def _pad_id(tok) -> int:
    return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def _batches(seqs: Sequence[tuple[list[int], int]], pad_id: int, batch_size: int) -> Iterator[tuple]:
    """Left-padded batches, longest first. Each (ids, start) scores ids[start:], which with left padding
    are the last n = len(ids) - start tokens of the row; logits are kept for the last k = max(n) + 1
    positions only. Yields (indices, input_ids, attention_mask, position_ids, n per row, k)."""
    for ids, start in seqs:
        if not 1 <= start < len(ids):
            raise ValueError("each sequence needs at least one context token and one scored token")
    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i][0]))
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        rows = [seqs[i][0] for i in idx]
        T = max(map(len, rows))
        ids = torch.full((len(rows), T), pad_id, dtype=torch.long)
        att = torch.zeros_like(ids)
        for b, r in enumerate(rows):
            ids[b, T - len(r):] = torch.tensor(r)
            att[b, T - len(r):] = 1
        n = [len(seqs[i][0]) - seqs[i][1] for i in idx]
        yield idx, ids, att, (att.cumsum(-1) - 1).clamp_min(0), n, max(n) + 1


def _tail_logprobs(model, ids, att, pos, k: int) -> torch.Tensor:
    """float32 log-softmax at the last k positions but one: row j predicts token T-k+1+j."""
    dev = next(model.parameters()).device
    logits = model(input_ids=ids.to(dev), attention_mask=att.to(dev), position_ids=pos.to(dev),
                   logits_to_keep=k).logits
    return torch.log_softmax(logits[:, :-1].float(), dim=-1)


@torch.no_grad()
def token_logprobs(model, seqs: Sequence[tuple[list[int], int]], pad_id: int,
                   batch_size: int = 4) -> list[torch.Tensor]:
    """Teacher forcing: for each (ids, start), log p(ids[t] | ids[:t]) for t = start .. len-1."""
    out: list[torch.Tensor] = [torch.empty(0)] * len(seqs)
    for idx, ids, att, pos, n, k in _batches(seqs, pad_id, batch_size):
        logp = _tail_logprobs(model, ids, att, pos, k)
        targets = ids[:, ids.shape[1] - k + 1:].to(logp.device)
        lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        for row, i in enumerate(idx):
            out[i] = lp[row, lp.shape[1] - n[row]:].cpu()
    return out


def _encode_pairs(tok, pairs: Sequence[Mapping], max_len: int | None) -> tuple[list, list, int]:
    """(ids, response_start) per pair via pencil.chat; pairs that do not fit max_len are skipped."""
    offset = chat.user_content_offset(tok)
    seqs, kept, skipped = [], [], 0
    for r in pairs:
        try:
            e = chat.encode_chat(tok, r["prompt"], r["answer"], max_len=max_len, user_offset=offset)
        except ValueError:
            skipped += 1
            continue
        seqs.append((e.input_ids, e.response_start))
        kept.append(r)
    return seqs, kept, skipped


def task_nll(model, tok, pairs: Sequence[Mapping], *, batch_size: int = 4, max_len: int | None = None) -> dict:
    """Per-token NLL of each answer given its prompt, on response tokens only (the loss positions of
    pencil.chat.encode_chat: the response and the template's end-of-turn tokens)."""
    seqs, kept, skipped = _encode_pairs(tok, pairs, max_len)
    lps = token_logprobs(model, seqs, _pad_id(tok), batch_size)
    items = [{"id": r.get("id"), "nll": float(-lp.sum()), "n_tokens": int(lp.numel())} for r, lp in zip(kept, lps)]
    total = sum(i["n_tokens"] for i in items)
    return {"nll_per_token": sum(i["nll"] for i in items) / total if total else None,
            "n_tokens": total, "n_items": len(items), "n_skipped": skipped, "items": items}


def mmlu_prompt(item: Mapping) -> str:
    options = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(item["choices"]))
    return f"{item['question']}\n\n{options}\n\nAnswer with the letter of the correct option."


def _answer_index(answer: Any) -> int:
    return LETTERS.index(answer.strip().upper()) if isinstance(answer, str) else int(answer)


def mmlu_scores(model, tok, items: Sequence[Mapping], *, batch_size: int = 8) -> list[dict]:
    """Log-likelihood multiple choice: the question goes through the chat template as one user turn and each
    letter is scored as the start of the response; the prediction is the most likely letter."""
    seqs = []
    for it in items:
        p = prompt_ids(tok, mmlu_prompt(it))
        seqs += [(p + tok(LETTERS[c], add_special_tokens=False)["input_ids"], len(p)) for c in range(len(it["choices"]))]
    lps = token_logprobs(model, seqs, _pad_id(tok), batch_size)
    out, at = [], 0
    for it in items:
        scores = [float(lp.sum()) for lp in lps[at:at + len(it["choices"])]]
        at += len(it["choices"])
        pred, ans = max(range(len(scores)), key=scores.__getitem__), _answer_index(it["answer"])
        out.append({"id": it.get("id"), "pred": pred, "answer": ans, "correct": pred == ans})
    return out


@torch.no_grad()
def response_kl(model, reference, tok, pairs: Sequence[Mapping], *, batch_size: int = 2,
                max_len: int | None = None) -> dict:
    """Mean per-token KL(p_reference || p_model) over the positions that predict response tokens, teacher-forced
    on the given (prompt, answer) pairs (evaluate() uses the reference's own greedy responses)."""
    seqs, kept, skipped = _encode_pairs(tok, pairs, max_len)
    kl_sum, n_tok = 0.0, 0
    for idx, ids, att, pos, n, k in _batches(seqs, _pad_id(tok), batch_size):
        lq = _tail_logprobs(model, ids, att, pos, k)
        lp = _tail_logprobs(reference, ids, att, pos, k).to(lq.device)
        kl = (lp.exp() * (lp - lq)).sum(-1)
        for row in range(len(idx)):
            kl_sum += float(kl[row, kl.shape[1] - n[row]:].sum())
            n_tok += n[row]
    return {"kl_per_token": kl_sum / n_tok if n_tok else None, "n_tokens": n_tok, "n_prompts": len(kept),
            "n_skipped": skipped}


def load_causal_lm(path: str | os.PathLike, *, dtype: torch.dtype | None = None, device: str | None = None):
    """A checkpoint (full model or PEFT adapter folder, local or hub) for teacher-forced metrics."""
    from transformers import AutoModelForCausalLM

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(str(path).removeprefix("hf:"), dtype=dtype or default_dtype())
    return model.to(device).eval()


# ------------------------------------------------------------------------------------ aggregation

def _wilson(k: int, n: int) -> tuple[float | None, float | None, float | None]:
    try:
        from .stats import wilson_ci
        vals = wilson_ci(k, n)
    except ImportError:                       # stats.py not there yet: rate only (k and n are stored)
        vals = (k / n if n else None, None, None)
    return tuple(None if v is None or math.isnan(v) else float(v) for v in vals)  # type: ignore[return-value]


def rate(flags: Sequence[bool | None]) -> dict:
    """k/n with a Wilson 95% interval; None entries (no judge verdict) are left out and counted."""
    vals = [bool(f) for f in flags if f is not None]
    p, lo, hi = _wilson(sum(vals), len(vals))
    return {"k": sum(vals), "n": len(vals), "p": p, "lo": lo, "hi": hi, "n_unlabelled": len(flags) - len(vals)}


def _is(label: str | None, *wanted: str) -> bool | None:
    return None if label is None else label in wanted


# ------------------------------------------------------------------------------------------ main

def _model_id(model: Any, given: str | None) -> str:
    if given:
        return given
    if isinstance(model, (str, os.PathLike)):
        return str(model)
    name = getattr(model, "name_or_path", None) or getattr(getattr(model, "config", None), "_name_or_path", None)
    if not name:
        raise ValueError("pass checkpoint_id / reference_id for an in-memory model")
    return str(name)


def _judge_spec(given: Any, hub_id: str | None, default: str) -> Any:
    """An explicit judge (spec, model object or False) wins; else the hub id in cfg["models"]; else the default."""
    if given is not None:
        return given
    return "hf:" + hub_id if hub_id else default


def _generations(model: Any, tok, splits: dict[str, list], name: str, cache: bool, root: Any,
                 gen_kw: dict) -> dict[str, dict[str, str]]:
    splits = {s: rows for s, rows in splits.items() if rows}
    if not splits:
        return {}
    if cache:
        return generate_cached(model, tok, splits, checkpoint=name, root=root, **gen_kw)
    flat = [(s, r) for s, rows in splits.items() for r in rows]
    texts = generate(model, tok, [r["prompt"] for _, r in flat], **gen_kw)
    out: dict[str, dict[str, str]] = {s: {} for s in splits}
    for (s, r), t in zip(flat, texts):
        out[s][r["id"]] = t
    return out


def _teacher_forced(checkpoint: Any, reference: Any, tok, sel: dict, ref_gens: dict | None, split: str,
                    ev: Mapping) -> dict:
    """Task NLL, MMLU and KL, loading a path checkpoint (and reference) once. Freed when this returns."""
    if not (sel["task"] or any(sel["mmlu"].values()) or ref_gens):
        return {"task": None, "mmlu": {}, "kl": {}}
    model = load_causal_lm(checkpoint) if isinstance(checkpoint, (str, os.PathLike)) else checkpoint
    bs, max_len = ev.get("tf_batch_size", 4), ev.get("max_len")
    out: dict[str, Any] = {
        "task": task_nll(model, tok, sel["task"], batch_size=bs, max_len=max_len) if sel["task"] else None,
        "mmlu": {l: mmlu_scores(model, tok, rows, batch_size=bs) for l, rows in sel["mmlu"].items() if rows},
        "kl": {},
    }
    if ref_gens:
        ref = load_causal_lm(reference) if isinstance(reference, (str, os.PathLike)) else reference
        for l, rows in sel["xstest"].items():
            gens = ref_gens.get(f"{split}_xstest_{l}", {})
            pairs = [{"id": r["id"], "prompt": r["prompt"], "answer": gens[r["id"]]} for r in rows if r["id"] in gens]
            if pairs:
                out["kl"][l] = response_kl(model, ref, tok, pairs, batch_size=ev.get("kl_batch_size", 2),
                                           max_len=max_len)
    return out


def evaluate(checkpoint: Any, tier: str, *, cfg: Mapping[str, Any], data: Mapping[str, Any],
             judges: Mapping[str, Any] | None, out_name: str, tokenizer=None, reference: Any = None,
             checkpoint_id: str | None = None, reference_id: str | None = None, cache: bool = False,
             gen_root: Any = None, out_dir: Any = None) -> dict:
    """Evaluate `checkpoint` (a path / hub id / PEFT folder, or an in-memory model with `checkpoint_id`).

    data: {"harmful": {lang: [{"id", "prompt"}]}, "xstest": {lang: [...]},
           "task": [{"id", "prompt", "answer"}], "mmlu": {lang: [{"id", "question", "choices", "answer"}]}}
    judges: {"harm": ..., "refusal": ...}, each None (the judge in cfg["models"]), an AuditKIT spec, an
            injected judge model (see pencil.judge), or False to skip that judge.
    reference: optional model or path for the KL metric. tokenizer: the pinned tokenizer (default: θ0's).
    cache: store / reuse raw generations in outputs/generations/ (gitignored; gen_root overrides).
    Writes outputs/results/{out_name}.json (out_dir overrides) and returns the same payload.
    """
    ev, gen_cfg, models = cfg.get("evaluation", {}), cfg.get("generation", {}), cfg.get("models", {})
    sel = select_tier(data, tier, cfg)
    split = TIER_SPLIT[tier]
    ckpt_id = _model_id(checkpoint, checkpoint_id)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(models["aligned"])
    gen_kw = {"max_new_tokens": gen_cfg.get("max_new_tokens", 256), "temperature": gen_cfg.get("temperature", 0.0),
              "batch_size": gen_cfg.get("batch_size", 16)}

    # 1. generation
    splits = {f"{split}_{kind}_{l}": rows for kind in ("harmful", "xstest") for l, rows in sel[kind].items()}
    gens = _generations(checkpoint, tokenizer, splits, ckpt_id, cache, gen_root, gen_kw)
    ref_id = _model_id(reference, reference_id) if reference is not None else None
    ref_gens = None
    if reference is not None:
        ref_splits = {f"{split}_xstest_{l}": rows for l, rows in sel["xstest"].items()}
        ref_gens = _generations(reference, tokenizer, ref_splits, ref_id, cache, gen_root, gen_kw)

    # 2. teacher-forced metrics
    tf = _teacher_forced(checkpoint, reference, tokenizer, sel, ref_gens, split, ev)
    free_gpu()

    # 3-4. judges, one at a time
    judges = dict(judges or {})
    harm = _judge_spec(judges.get("harm"), models.get("harm_judge"), DEFAULT_HARM_JUDGE)
    refj = _judge_spec(judges.get("refusal"), models.get("refusal_judge"), DEFAULT_REFUSAL_JUDGE)
    rubric = ev.get("refusal_rubric", REFUSAL_RUBRIC)
    bs = ev.get("judge_batch_size", 8)
    H = [(l, r, gens[f"{split}_harmful_{l}"][r["id"]]) for l, rows in sel["harmful"].items() for r in rows]
    X = [(l, r, gens[f"{split}_xstest_{l}"][r["id"]]) for l, rows in sel["xstest"].items() for r in rows]
    guard = (harm_judge([r["prompt"] for _, r, _ in H], [c for *_, c in H], model=harm, batch_size=bs)
             if harm is not False else [{"unsafe": None, "categories": []} for _ in H])
    both = H + X
    refusal = (refusal_judge([r["prompt"] for _, r, _ in both], [c for *_, c in both], [l for l, *_ in both],
                             model=refj, rubric_path=rubric, batch_size=bs)
               if refj is not False else [{"label": None} for _ in both])
    ref_h, ref_x = refusal[:len(H)], refusal[len(H):]

    # aggregate: labels, ids and hashes only
    items = {
        "harmful": [{"id": r["id"], "lang": l, "completion_sha256": sha256(c), "refusal": f["label"],
                     "unsafe": g["unsafe"], "categories": g["categories"]}
                    for (l, r, c), g, f in zip(H, guard, ref_h)],
        "xstest": [{"id": r["id"], "lang": l, "completion_sha256": sha256(c), "refusal": f["label"]}
                   for (l, r, c), f in zip(X, ref_x)],
        "mmlu": [{"lang": l, **s} for l, scores in tf["mmlu"].items() for s in scores],
        "task": tf["task"]["items"] if tf["task"] else [],
    }
    order = list(cfg.get("languages", []))
    langs = sorted({*sel["harmful"], *sel["xstest"], *sel["mmlu"]},
                   key=lambda l: (order.index(l) if l in order else len(order), l))
    metrics: dict[str, dict] = {}
    for l in langs:
        h = [i for i in items["harmful"] if i["lang"] == l]
        x = [i for i in items["xstest"] if i["lang"] == l]
        m: dict[str, Any] = {}
        if h:
            m["harmful"] = {"n_prompts": len(h), "refusal": rate([_is(i["refusal"], "REFUSE") for i in h]),
                            "partial": rate([_is(i["refusal"], "PARTIAL") for i in h]),
                            "unsafe": rate([i["unsafe"] for i in h])}
        if x:
            m["xstest"] = {"n_prompts": len(x), "over_refusal": rate([_is(i["refusal"], "REFUSE") for i in x]),
                           "over_refusal_or_partial": rate([_is(i["refusal"], "REFUSE", "PARTIAL") for i in x])}
        if l in tf["mmlu"]:
            m["mmlu"] = {"accuracy": rate([s["correct"] for s in tf["mmlu"][l]])}
        if l in tf["kl"]:
            m["kl_to_reference"] = tf["kl"][l]
        metrics[l] = m
    payload = {
        "checkpoint": ckpt_id, "tier": tier, "data_split": split, "langs": langs, "reference": ref_id,
        "generation": {k: gen_kw[k] for k in ("max_new_tokens", "temperature")},
        "judges": {"harm": None if harm is False else judge_name(harm, DEFAULT_HARM_JUDGE),
                   "refusal": None if refj is False else judge_name(refj, DEFAULT_REFUSAL_JUDGE),
                   "refusal_rubric_sha256": sha256(read_rubric(rubric))},
        "metrics": metrics,
        "task": {k: v for k, v in tf["task"].items() if k != "items"} if tf["task"] else None,
        "items": items,
    }
    write_result(out_name, payload, config=dict(cfg), seed=cfg.get("split_seed"), checkpoint=ckpt_id,
                 out_dir=out_dir)
    return payload
