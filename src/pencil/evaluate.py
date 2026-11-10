"""Per-language evaluation of one checkpoint.

Per language: refusal and harmful-compliance (Llama Guard "unsafe") rates on harmful prompts and the
over-refusal rate on XSTest-derived benign prompts, each with a Wilson 95% interval; Global-MMLU
log-likelihood accuracy; and, with a reference model, mean per-token KL(p_reference || p_checkpoint) on the
benign prompts. Plus the per-token NLL of held-out task answers.

The steps run in this order so that a checkpoint given as a path never shares the GPU with a judge:
generate (cached in outputs/generations/ with cache=True); teacher-forced metrics; Llama Guard 3; the refusal
judge. An in-memory model stays loaded throughout (its owner frees it), so pass a path for GPU runs.

Results hold ids, labels and sha256 hashes of completions, never completion text (CLAUDE.md rule 7).
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch

from . import chat
from .generate import (
    default_dtype,
    free_gpu,
    generate,
    generate_cached,
    longest_first_batches,
    pad_id,
    prompt_ids,
    sha256,
)
from .judge import (
    DEFAULT_HARM_JUDGE,
    DEFAULT_REFUSAL_JUDGE,
    REFUSAL_RUBRIC,
    harm_judge,
    judge_name,
    read_rubric,
    refusal_judge,
)
from .results import write_result
from .stats import wilson_ci

TIER_SPLIT = {"probe": "dev", "full": "test"}  # CLAUDE.md rule 5
LETTERS = "ABCDEFGHIJ"


# ------------------------------------------------------------------------------- data selection


def select_tier(data: Mapping[str, Any], tier: str, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The items one tier evaluates. probe: the first `evaluation.probe_per_lang` items of every list;
    full: everything. MMLU is also capped at `evaluation.mmlu_per_lang`."""
    if tier not in TIER_SPLIT:
        raise ValueError(f"tier must be one of {sorted(TIER_SPLIT)}, got {tier!r}")
    ev = cfg.get("evaluation", {})
    cap = ev.get("probe_per_lang") if tier == "probe" else None
    mmlu_cap = min((c for c in (cap, ev.get("mmlu_per_lang")) if c), default=None)

    def take(rows: Sequence[Mapping], n: int | None, what: str) -> list:
        rows = list(rows)[:n] if n else list(rows)
        ids = [r.get("id") for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate or missing ids in {what}")
        return rows

    return {
        "harmful": {lang: take(r, cap, f"harmful/{lang}") for lang, r in data.get("harmful", {}).items()},
        "xstest": {lang: take(r, cap, f"xstest/{lang}") for lang, r in data.get("xstest", {}).items()},
        "task": take(data.get("task", []), cap, "task"),
        "mmlu": {lang: take(r, mmlu_cap, f"mmlu/{lang}") for lang, r in data.get("mmlu", {}).items()},
    }


# ---------------------------------------------------------------------------- teacher forcing


def _batches(seqs: Sequence[tuple[list[int], int]], pad: int, batch_size: int) -> Iterator[tuple]:
    """Left-padded batches of (ids, start) sequences, longest first. Each row scores ids[start:], which with
    left padding are its last n = len(ids) - start tokens.
    Yields (indices, input_ids, attention_mask, position_ids, n per row)."""
    for ids, start in seqs:
        if not 1 <= start < len(ids):
            raise ValueError("each sequence needs at least one context token and one scored token")
    for idx in longest_first_batches([len(ids) for ids, _ in seqs], batch_size):
        batch = chat.collate([chat.Encoded(seqs[i][0], 0, seqs[i][1]) for i in idx], pad, side="left")
        att = batch["attention_mask"]
        n = [len(seqs[i][0]) - seqs[i][1] for i in idx]
        yield idx, batch["input_ids"], att, (att.cumsum(-1) - 1).clamp_min(0), n


def _tail_logprobs(model, ids, att, pos, n: int) -> torch.Tensor:
    """float32 log-softmax over the predictions of the last n tokens: column j predicts token T - n + j."""
    dev = next(model.parameters()).device
    logits = model(
        input_ids=ids.to(dev), attention_mask=att.to(dev), position_ids=pos.to(dev), logits_to_keep=n + 1
    ).logits
    return torch.log_softmax(logits[:, :-1].float(), dim=-1)


@torch.no_grad()
def token_logprobs(model, seqs: Sequence[tuple[list[int], int]], pad: int, batch_size: int = 4) -> list[torch.Tensor]:
    """Teacher forcing: for each (ids, start), log p(ids[t] | ids[:t]) for t = start .. len-1."""
    out: list[torch.Tensor] = [torch.empty(0)] * len(seqs)
    for idx, ids, att, pos, n in _batches(seqs, pad, batch_size):
        logp = _tail_logprobs(model, ids, att, pos, max(n))
        targets = ids[:, -max(n) :].to(logp.device)
        lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        for row, i in enumerate(idx):
            out[i] = lp[row, -n[row] :].cpu()
    return out


def _encode_pairs(tok, pairs: Sequence[Mapping], max_len: int | None) -> tuple[list, list, int]:
    """(ids, response_start) per pair via pencil.chat, the pairs kept, and how many were skipped because
    they do not fit max_len."""
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
    """Per-token NLL of each answer given its prompt, on the loss positions of pencil.chat.encode_chat (the
    response and the template's end-of-turn tokens)."""
    seqs, kept, skipped = _encode_pairs(tok, pairs, max_len)
    lps = token_logprobs(model, seqs, pad_id(tok), batch_size)
    items = [{"id": r.get("id"), "nll": float(-lp.sum()), "n_tokens": int(lp.numel())} for r, lp in zip(kept, lps)]
    total = sum(i["n_tokens"] for i in items)
    return {
        "nll_per_token": sum(i["nll"] for i in items) / total if total else None,
        "n_tokens": total,
        "n_items": len(items),
        "n_skipped": skipped,
        "items": items,
    }


def mmlu_prompt(item: Mapping) -> str:
    options = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(item["choices"]))
    return f"{item['question']}\n\n{options}\n\nAnswer with the letter of the correct option."


def _answer_index(answer: Any) -> int:
    return LETTERS.index(answer.strip().upper()) if isinstance(answer, str) else int(answer)


def mmlu_scores(model, tok, items: Sequence[Mapping], *, batch_size: int = 8) -> list[dict]:
    """Log-likelihood multiple choice: the question is one templated user turn, each option letter is scored
    as the start of the response, and the most likely letter is the prediction."""
    seqs = []
    for it in items:
        p = prompt_ids(tok, mmlu_prompt(it))
        seqs += [
            (p + tok(letter, add_special_tokens=False)["input_ids"], len(p)) for letter in LETTERS[: len(it["choices"])]
        ]
    lps = iter(token_logprobs(model, seqs, pad_id(tok), batch_size))
    out = []
    for it in items:
        scores = [float(next(lps).sum()) for _ in it["choices"]]
        pred, answer = max(range(len(scores)), key=scores.__getitem__), _answer_index(it["answer"])
        out.append({"id": it.get("id"), "pred": pred, "answer": answer, "correct": pred == answer})
    return out


@torch.no_grad()
def response_kl(
    model, reference, tok, pairs: Sequence[Mapping], *, batch_size: int = 2, max_len: int | None = None
) -> dict:
    """Mean per-token KL(p_reference || p_model) over the positions predicting response tokens, teacher-forced
    on the given (prompt, answer) pairs (evaluate() passes the reference's own greedy responses)."""
    seqs, kept, skipped = _encode_pairs(tok, pairs, max_len)
    kl_sum, n_tok = 0.0, 0
    for _, ids, att, pos, n in _batches(seqs, pad_id(tok), batch_size):
        lq = _tail_logprobs(model, ids, att, pos, max(n))
        lp = _tail_logprobs(reference, ids, att, pos, max(n)).to(lq.device)
        kl = (lp.exp() * (lp - lq)).sum(-1)
        for row, n_row in enumerate(n):
            kl_sum += float(kl[row, -n_row:].sum())
            n_tok += n_row
    return {
        "kl_per_token": kl_sum / n_tok if n_tok else None,
        "n_tokens": n_tok,
        "n_prompts": len(kept),
        "n_skipped": skipped,
    }


def _as_model(model_or_path: Any):
    """An in-memory model as is; a path / hub id / PEFT folder loaded for teacher forcing."""
    if not isinstance(model_or_path, (str, os.PathLike)):
        return model_or_path
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(str(model_or_path).removeprefix("hf:"), dtype=default_dtype())
    return model.to(device).eval()


def _teacher_forced(
    checkpoint: Any, reference: Any, tok, sel: dict, ref_gens: dict | None, split: str, ev: Mapping
) -> dict:
    """Task NLL, MMLU and KL to the reference, loading a path checkpoint (and reference) once."""
    if not (sel["task"] or any(sel["mmlu"].values()) or ref_gens):
        return {"task": None, "mmlu": {}, "kl": {}}
    model = _as_model(checkpoint)
    bs, max_len = ev.get("tf_batch_size", 4), ev.get("max_len")
    out: dict[str, Any] = {
        "task": task_nll(model, tok, sel["task"], batch_size=bs, max_len=max_len) if sel["task"] else None,
        "mmlu": {lang: mmlu_scores(model, tok, rows, batch_size=bs) for lang, rows in sel["mmlu"].items() if rows},
        "kl": {},
    }
    if ref_gens:
        ref = _as_model(reference)
        for lang, rows in sel["xstest"].items():
            gens = ref_gens.get(f"{split}_xstest_{lang}", {})
            pairs = [{"id": r["id"], "prompt": r["prompt"], "answer": gens[r["id"]]} for r in rows if r["id"] in gens]
            if pairs:
                out["kl"][lang] = response_kl(
                    model, ref, tok, pairs, batch_size=ev.get("kl_batch_size", 2), max_len=max_len
                )
    return out


# ------------------------------------------------------------------------------------ aggregation


def rate(flags: Sequence[bool | None]) -> dict:
    """k/n with a Wilson 95% interval; None entries (no judge verdict) are left out and counted."""
    vals = [bool(f) for f in flags if f is not None]
    p, lo, hi = (None if math.isnan(v) else v for v in wilson_ci(sum(vals), len(vals)))
    return {"k": sum(vals), "n": len(vals), "p": p, "lo": lo, "hi": hi, "n_unlabelled": len(flags) - len(vals)}


def _label_in(label: str | None, *wanted: str) -> bool | None:
    return None if label is None else label in wanted


def _lang_metrics(lang: str, items: dict, tf: dict) -> dict:
    """The metrics block of one language."""
    h = [i for i in items["harmful"] if i["lang"] == lang]
    x = [i for i in items["xstest"] if i["lang"] == lang]
    m: dict[str, Any] = {}
    if h:
        m["harmful"] = {
            "n_prompts": len(h),
            "refusal": rate([_label_in(i["refusal"], "REFUSE") for i in h]),
            "partial": rate([_label_in(i["refusal"], "PARTIAL") for i in h]),
            "unsafe": rate([i["unsafe"] for i in h]),
        }
    if x:
        m["xstest"] = {
            "n_prompts": len(x),
            "over_refusal": rate([_label_in(i["refusal"], "REFUSE") for i in x]),
            "over_refusal_or_partial": rate([_label_in(i["refusal"], "REFUSE", "PARTIAL") for i in x]),
        }
    if lang in tf["mmlu"]:
        m["mmlu"] = {"accuracy": rate([s["correct"] for s in tf["mmlu"][lang]])}
    if lang in tf["kl"]:
        m["kl_to_reference"] = tf["kl"][lang]
    return m


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


def _generations(
    model: Any, tok, splits: dict[str, list], name: str, cache: bool, root: Any, gen_kw: dict
) -> dict[str, dict[str, str]]:
    """{split: {id: completion}} for the non-empty splits, through the cache when `cache` is set."""
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


def _judge_all(harmful: list, benign: list, harm: Any, refj: Any, rubric: str, batch_size: int) -> tuple:
    """Llama Guard on the harmful responses, then the refusal judge on harmful + benign ones; a judge set to
    False gives empty labels. Returns (guard labels, refusal labels on harmful, refusal labels on benign)."""
    if harm is False:
        guard = [{"unsafe": None, "categories": []} for _ in harmful]
    else:
        guard = harm_judge(
            [r["prompt"] for _, r, _ in harmful], [c for *_, c in harmful], model=harm, batch_size=batch_size
        )
    both = harmful + benign
    if refj is False:
        refusal = [{"label": None} for _ in both]
    else:
        refusal = refusal_judge(
            [r["prompt"] for _, r, _ in both],
            [c for *_, c in both],
            [lang for lang, *_ in both],
            model=refj,
            rubric_path=rubric,
            batch_size=batch_size,
        )
    return guard, refusal[: len(harmful)], refusal[len(harmful) :]


def evaluate(
    checkpoint: Any,
    tier: str,
    *,
    cfg: Mapping[str, Any],
    data: Mapping[str, Any],
    judges: Mapping[str, Any] | None,
    out_name: str,
    tokenizer=None,
    reference: Any = None,
    checkpoint_id: str | None = None,
    reference_id: str | None = None,
    cache: bool = False,
    gen_root: Any = None,
    out_dir: Any = None,
) -> dict:
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
    judges = judges or {}
    sel = select_tier(data, tier, cfg)
    split = TIER_SPLIT[tier]
    ckpt_id = _model_id(checkpoint, checkpoint_id)
    ref_id = _model_id(reference, reference_id) if reference is not None else None
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(models["aligned"])
    gen_kw = {
        "max_new_tokens": gen_cfg.get("max_new_tokens", 256),
        "temperature": gen_cfg.get("temperature", 0.0),
        "batch_size": gen_cfg.get("batch_size", 16),
    }

    # 1. generation
    splits = {f"{split}_{kind}_{lang}": rows for kind in ("harmful", "xstest") for lang, rows in sel[kind].items()}
    gens = _generations(checkpoint, tokenizer, splits, ckpt_id, cache, gen_root, gen_kw)
    ref_gens = None
    if reference is not None:
        ref_splits = {f"{split}_xstest_{lang}": rows for lang, rows in sel["xstest"].items()}
        ref_gens = _generations(reference, tokenizer, ref_splits, ref_id, cache, gen_root, gen_kw)

    # 2. teacher-forced metrics
    tf = _teacher_forced(checkpoint, reference, tokenizer, sel, ref_gens, split, ev)
    free_gpu()

    # 3-4. judges, one at a time, on (lang, row, completion) triples
    harmful = [
        (lang, r, gens[f"{split}_harmful_{lang}"][r["id"]]) for lang, rows in sel["harmful"].items() for r in rows
    ]
    benign = [(lang, r, gens[f"{split}_xstest_{lang}"][r["id"]]) for lang, rows in sel["xstest"].items() for r in rows]
    harm = _judge_spec(judges.get("harm"), models.get("harm_judge"), DEFAULT_HARM_JUDGE)
    refj = _judge_spec(judges.get("refusal"), models.get("refusal_judge"), DEFAULT_REFUSAL_JUDGE)
    rubric = ev.get("refusal_rubric", REFUSAL_RUBRIC)
    guard, ref_h, ref_x = _judge_all(harmful, benign, harm, refj, rubric, ev.get("judge_batch_size", 8))

    # results: labels, ids and hashes only
    items = {
        "harmful": [
            {
                "id": r["id"],
                "lang": lang,
                "completion_sha256": sha256(c),
                "refusal": f["label"],
                "unsafe": g["unsafe"],
                "categories": g["categories"],
            }
            for (lang, r, c), g, f in zip(harmful, guard, ref_h)
        ],
        "xstest": [
            {"id": r["id"], "lang": lang, "completion_sha256": sha256(c), "refusal": f["label"]}
            for (lang, r, c), f in zip(benign, ref_x)
        ],
        "mmlu": [{"lang": lang, **s} for lang, scores in tf["mmlu"].items() for s in scores],
        "task": tf["task"]["items"] if tf["task"] else [],
    }
    order = list(cfg.get("languages", []))
    langs = sorted(
        {*sel["harmful"], *sel["xstest"], *sel["mmlu"]},
        key=lambda lang: (order.index(lang) if lang in order else len(order), lang),
    )
    payload = {
        "checkpoint": ckpt_id,
        "tier": tier,
        "data_split": split,
        "langs": langs,
        "reference": ref_id,
        "generation": {k: gen_kw[k] for k in ("max_new_tokens", "temperature")},
        "judges": {
            "harm": None if harm is False else judge_name(harm, DEFAULT_HARM_JUDGE),
            "refusal": None if refj is False else judge_name(refj, DEFAULT_REFUSAL_JUDGE),
            "refusal_rubric_sha256": sha256(read_rubric(rubric)),
        },
        "metrics": {lang: _lang_metrics(lang, items, tf) for lang in langs},
        "task": {k: v for k, v in tf["task"].items() if k != "items"} if tf["task"] else None,
        "items": items,
    }
    write_result(out_name, payload, config=dict(cfg), seed=cfg.get("split_seed"), checkpoint=ckpt_id, out_dir=out_dir)
    return payload
