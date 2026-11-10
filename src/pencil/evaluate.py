"""Per-language evaluation of one checkpoint.

Per language: refusal and harmful-compliance (Llama Guard "unsafe") rates on harmful prompts and the
over-refusal rate on XSTest-derived benign prompts, each with a Wilson 95% interval; Global-MMLU
log-likelihood accuracy; and, with a reference model, mean per-token KL(p_reference || p_checkpoint) on the
benign prompts. Plus the per-token NLL of held-out task answers.

`evaluate` keeps one model on the GPU at a time, in this order (see its body):
  1. generate: a path checkpoint is loaded, run and unloaded inside `generate` (cached in outputs/generations/
     with cache=True);
  2. teacher-forced metrics: the checkpoint is loaded again, then released with `free_gpu`;
  3. Llama Guard 3: loaded, run and unloaded inside `harm_judge`;
  4. the Aya Expanse refusal judge: loaded, run and unloaded inside `refusal_judge`.
An in-memory model stays loaded throughout (its owner frees it), so pass a path for GPU runs.

Results hold ids, labels and sha256 hashes of completions, never completion text (CLAUDE.md rule 7).
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, NamedTuple

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


def _take_rows(rows: Sequence[Mapping], limit: int | None, what: str) -> list:
    """The first `limit` rows (all of them when limit is unset); raises on duplicate or missing ids."""
    rows = list(rows)
    if limit:
        rows = rows[:limit]
    ids = [row.get("id") for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate or missing ids in {what}")
    return rows


def _take_per_lang(rows_by_lang: Mapping[str, Sequence[Mapping]], limit: int | None, kind: str) -> dict:
    """`_take_rows` for each language's list."""
    return {lang: _take_rows(rows, limit, f"{kind}/{lang}") for lang, rows in rows_by_lang.items()}


def _smallest_cap(*caps: int | None) -> int | None:
    """The smallest of the caps that are set, or None when none is."""
    set_caps = [cap for cap in caps if cap]
    return min(set_caps, default=None)


def select_tier(data: Mapping[str, Any], tier: str, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The items one tier evaluates. probe: the first `evaluation.probe_per_lang` items of every list;
    full: everything. MMLU is also capped at `evaluation.mmlu_per_lang`."""
    if tier not in TIER_SPLIT:
        raise ValueError(f"tier must be one of {sorted(TIER_SPLIT)}, got {tier!r}")
    eval_cfg = cfg.get("evaluation", {})
    cap = eval_cfg.get("probe_per_lang") if tier == "probe" else None  # the full tier is never capped
    mmlu_cap = _smallest_cap(cap, eval_cfg.get("mmlu_per_lang"))
    return {
        "harmful": _take_per_lang(data.get("harmful", {}), cap, "harmful"),
        "xstest": _take_per_lang(data.get("xstest", {}), cap, "xstest"),
        "task": _take_rows(data.get("task", []), cap, "task"),
        "mmlu": _take_per_lang(data.get("mmlu", {}), mmlu_cap, "mmlu"),
    }


def _split_key(split: str, kind: str, lang: str) -> str:
    """The generation-cache split name, e.g. "dev_harmful_hi" (scripts/02_label_sample.py reads these)."""
    return f"{split}_{kind}_{lang}"


# ---------------------------------------------------------------------------- teacher forcing


class _Batch(NamedTuple):
    """A left-padded batch of (ids, start) sequences; row r scores its last n_scored[r] tokens."""

    indices: list[int]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    n_scored: list[int]


def _left_padded_batches(sequences: Sequence[tuple[list[int], int]], pad: int, batch_size: int) -> Iterator[_Batch]:
    """Left-padded batches of (ids, start) sequences, longest first. Each row scores ids[start:], which with
    left padding are its last n = len(ids) - start tokens."""
    for ids, start in sequences:
        if not 1 <= start < len(ids):
            raise ValueError("each sequence needs at least one context token and one scored token")
    lengths = [len(ids) for ids, _ in sequences]
    for indices in longest_first_batches(lengths, batch_size):
        encoded = [chat.Encoded(sequences[i][0], 0, sequences[i][1]) for i in indices]
        padded = chat.collate(encoded, pad, side="left")
        attention_mask = padded["attention_mask"]
        position_ids = (attention_mask.cumsum(-1) - 1).clamp_min(0)  # positions restart at 0 after the padding
        n_scored = [len(sequences[i][0]) - sequences[i][1] for i in indices]
        yield _Batch(indices, padded["input_ids"], attention_mask, position_ids, n_scored)


def _tail_logprobs(model, batch: _Batch, n: int) -> torch.Tensor:
    """float32 log-softmax over the predictions of the last n tokens: column j predicts token T - n + j."""
    device = next(model.parameters()).device
    output = model(
        input_ids=batch.input_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        position_ids=batch.position_ids.to(device),
        logits_to_keep=n + 1,
    )
    return torch.log_softmax(output.logits[:, :-1].float(), dim=-1)


@torch.no_grad()
def token_logprobs(
    model, sequences: Sequence[tuple[list[int], int]], pad: int, batch_size: int = 4
) -> list[torch.Tensor]:
    """Teacher forcing: for each (ids, start), log p(ids[t] | ids[:t]) for t = start .. len-1."""
    logprobs_per_sequence: list[torch.Tensor] = [torch.empty(0)] * len(sequences)
    for batch in _left_padded_batches(sequences, pad, batch_size):
        n_max = max(batch.n_scored)
        logprobs = _tail_logprobs(model, batch, n_max)
        targets = batch.input_ids[:, -n_max:].to(logprobs.device)
        target_logprobs = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        for row, i in enumerate(batch.indices):
            logprobs_per_sequence[i] = target_logprobs[row, -batch.n_scored[row] :].cpu()
    return logprobs_per_sequence


def _encode_pairs(tokenizer, pairs: Sequence[Mapping], max_len: int | None) -> tuple[list, list, int]:
    """(ids, response_start) per pair via pencil.chat, the pairs kept, and how many were skipped because
    they do not fit max_len."""
    user_offset = chat.user_content_offset(tokenizer)
    sequences, kept, n_skipped = [], [], 0
    for pair in pairs:
        try:
            encoded = chat.encode_chat(
                tokenizer, pair["prompt"], pair["answer"], max_len=max_len, user_offset=user_offset
            )
        except ValueError:
            n_skipped += 1
            continue
        sequences.append((encoded.input_ids, encoded.response_start))
        kept.append(pair)
    return sequences, kept, n_skipped


def _nll_item(pair: Mapping, answer_logprobs: torch.Tensor) -> dict:
    """One task item's id, total NLL of its answer tokens, and how many tokens that is."""
    return {"id": pair.get("id"), "nll": float(-answer_logprobs.sum()), "n_tokens": int(answer_logprobs.numel())}


def task_nll(model, tokenizer, pairs: Sequence[Mapping], *, batch_size: int = 4, max_len: int | None = None) -> dict:
    """Per-token NLL of each answer given its prompt, on the loss positions of pencil.chat.encode_chat (the
    response and the template's end-of-turn tokens)."""
    sequences, kept, n_skipped = _encode_pairs(tokenizer, pairs, max_len)
    logprobs = token_logprobs(model, sequences, pad_id(tokenizer), batch_size)
    items = [_nll_item(pair, answer_logprobs) for pair, answer_logprobs in zip(kept, logprobs)]
    total_tokens = sum(item["n_tokens"] for item in items)
    total_nll = sum(item["nll"] for item in items)
    nll_per_token = total_nll / total_tokens if total_tokens else None
    return {
        "nll_per_token": nll_per_token,
        "n_tokens": total_tokens,
        "n_items": len(items),
        "n_skipped": n_skipped,
        "items": items,
    }


def mmlu_prompt(item: Mapping) -> str:
    """The question, its options as "A. ...", "B. ..." lines, and an instruction to answer with a letter."""
    options = "\n".join(f"{LETTERS[i]}. {choice}" for i, choice in enumerate(item["choices"]))
    return f"{item['question']}\n\n{options}\n\nAnswer with the letter of the correct option."


def _answer_index(answer: Any) -> int:
    """The correct option's index, from a letter ("B") or a number (1)."""
    if isinstance(answer, str):
        return LETTERS.index(answer.strip().upper())
    return int(answer)


def mmlu_scores(model, tokenizer, items: Sequence[Mapping], *, batch_size: int = 8) -> list[dict]:
    """Log-likelihood multiple choice: the question is one templated user turn, each option letter is scored
    as the start of the response, and the most likely letter is the prediction."""
    sequences = []
    for item in items:
        question_ids = prompt_ids(tokenizer, mmlu_prompt(item))
        for letter in LETTERS[: len(item["choices"])]:
            letter_ids = tokenizer(letter, add_special_tokens=False)["input_ids"]
            sequences.append((question_ids + letter_ids, len(question_ids)))
    letter_logprobs = iter(token_logprobs(model, sequences, pad_id(tokenizer), batch_size))

    scores = []
    for item in items:
        option_scores = [float(next(letter_logprobs).sum()) for _ in item["choices"]]
        prediction = max(range(len(option_scores)), key=option_scores.__getitem__)
        answer = _answer_index(item["answer"])
        scores.append({"id": item.get("id"), "pred": prediction, "answer": answer, "correct": prediction == answer})
    return scores


@torch.no_grad()
def response_kl(
    model, reference, tokenizer, pairs: Sequence[Mapping], *, batch_size: int = 2, max_len: int | None = None
) -> dict:
    """Mean per-token KL(p_reference || p_model) over the positions predicting response tokens, teacher-forced
    on the given (prompt, answer) pairs (evaluate() passes the reference's own greedy responses).

    KL(p ‖ q) = Σ_v p(v) (log p(v) − log q(v)) at each position, p the reference and q the model.
    """
    sequences, kept, n_skipped = _encode_pairs(tokenizer, pairs, max_len)
    kl_sum, n_tokens = 0.0, 0
    for batch in _left_padded_batches(sequences, pad_id(tokenizer), batch_size):
        n_max = max(batch.n_scored)
        model_logprobs = _tail_logprobs(model, batch, n_max)
        reference_logprobs = _tail_logprobs(reference, batch, n_max).to(model_logprobs.device)
        kl = (reference_logprobs.exp() * (reference_logprobs - model_logprobs)).sum(-1)
        for row, n_row in enumerate(batch.n_scored):
            kl_sum += float(kl[row, -n_row:].sum())
            n_tokens += n_row
    kl_per_token = kl_sum / n_tokens if n_tokens else None
    return {"kl_per_token": kl_per_token, "n_tokens": n_tokens, "n_prompts": len(kept), "n_skipped": n_skipped}


def _as_model(model_or_path: Any):
    """An in-memory model as is; a path / hub id / PEFT folder loaded for teacher forcing."""
    if not isinstance(model_or_path, (str, os.PathLike)):
        return model_or_path
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = str(model_or_path).removeprefix("hf:")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=default_dtype())
    return model.to(device).eval()


def _kl_to_reference(
    model, reference: Any, tokenizer, xstest: dict, reference_answers: dict, split: str, eval_cfg: Mapping
) -> dict:
    """Per language, `response_kl` of the checkpoint against the reference, on the reference's own answers to
    the benign prompts."""
    reference_model = _as_model(reference)
    kl_by_lang = {}
    for lang, rows in xstest.items():
        answers = reference_answers.get(_split_key(split, "xstest", lang), {})
        pairs = []
        for row in rows:
            if row["id"] in answers:
                pairs.append({"id": row["id"], "prompt": row["prompt"], "answer": answers[row["id"]]})
        if pairs:
            kl_by_lang[lang] = response_kl(
                model,
                reference_model,
                tokenizer,
                pairs,
                batch_size=eval_cfg.get("kl_batch_size", 2),
                max_len=eval_cfg.get("max_len"),
            )
    return kl_by_lang


def _teacher_forced(
    checkpoint: Any,
    reference: Any,
    tokenizer,
    selected: dict,
    reference_answers: dict | None,
    split: str,
    eval_cfg: Mapping,
) -> dict:
    """Task NLL, MMLU scores and KL to the reference, as {"task", "mmlu", "kl"}.

    A path checkpoint (and reference) is loaded once here. Nothing keeps it after this returns, so the caller's
    `free_gpu()` releases it before any judge is loaded.
    """
    has_task = bool(selected["task"])
    has_mmlu = any(selected["mmlu"].values())
    has_reference = bool(reference_answers)
    if not (has_task or has_mmlu or has_reference):
        return {"task": None, "mmlu": {}, "kl": {}}

    model = _as_model(checkpoint)
    batch_size = eval_cfg.get("tf_batch_size", 4)
    max_len = eval_cfg.get("max_len")
    task = None
    if has_task:
        task = task_nll(model, tokenizer, selected["task"], batch_size=batch_size, max_len=max_len)
    mmlu = {}
    for lang, rows in selected["mmlu"].items():
        if rows:
            mmlu[lang] = mmlu_scores(model, tokenizer, rows, batch_size=batch_size)
    kl = {}
    if has_reference:
        kl = _kl_to_reference(model, reference, tokenizer, selected["xstest"], reference_answers, split, eval_cfg)
    return {"task": task, "mmlu": mmlu, "kl": kl}


# ------------------------------------------------------------------------------------ aggregation


def _nan_to_none(value: float) -> float | None:
    """None for nan (JSON has no nan), else the value."""
    if math.isnan(value):
        return None
    return value


def rate(flags: Sequence[bool | None]) -> dict:
    """k/n with a Wilson 95% interval; None entries (no judge verdict) are left out and counted."""
    labelled = [bool(flag) for flag in flags if flag is not None]
    k = sum(labelled)
    n = len(labelled)
    p, lo, hi = [_nan_to_none(value) for value in wilson_ci(k, n)]
    return {"k": k, "n": n, "p": p, "lo": lo, "hi": hi, "n_unlabelled": len(flags) - n}


def _label_in(label: str | None, *wanted: str) -> bool | None:
    """Whether a judge label is one of `wanted`; None when there is no label."""
    if label is None:
        return None
    return label in wanted


def _lang_metrics(lang: str, items: dict, teacher_forced: dict) -> dict:
    """The metrics block of one language."""
    harmful = [item for item in items["harmful"] if item["lang"] == lang]
    benign = [item for item in items["xstest"] if item["lang"] == lang]
    metrics: dict[str, Any] = {}
    if harmful:
        metrics["harmful"] = {
            "n_prompts": len(harmful),
            "refusal": rate([_label_in(item["refusal"], "REFUSE") for item in harmful]),
            "partial": rate([_label_in(item["refusal"], "PARTIAL") for item in harmful]),
            "unsafe": rate([item["unsafe"] for item in harmful]),
        }
    if benign:
        metrics["xstest"] = {
            "n_prompts": len(benign),
            "over_refusal": rate([_label_in(item["refusal"], "REFUSE") for item in benign]),
            "over_refusal_or_partial": rate([_label_in(item["refusal"], "REFUSE", "PARTIAL") for item in benign]),
        }
    if lang in teacher_forced["mmlu"]:
        metrics["mmlu"] = {"accuracy": rate([score["correct"] for score in teacher_forced["mmlu"][lang]])}
    if lang in teacher_forced["kl"]:
        metrics["kl_to_reference"] = teacher_forced["kl"][lang]
    return metrics


def _ordered_langs(selected: dict, configured: Sequence[str]) -> list[str]:
    """Every language with harmful, benign or MMLU items: the configured ones in config order, then the rest
    alphabetically."""
    configured = list(configured)
    langs = {*selected["harmful"], *selected["xstest"], *selected["mmlu"]}

    def sort_key(lang: str) -> tuple[int, str]:
        if lang in configured:
            return configured.index(lang), lang
        return len(configured), lang

    return sorted(langs, key=sort_key)


# ------------------------------------------------------------------------------------------ main


class _Answered(NamedTuple):
    """One prompt and the checkpoint's completion of it, on its way to the judges. The completion text never
    leaves `evaluate`: results keep only its sha256."""

    lang: str
    row: Mapping[str, Any]
    completion: str


def _model_id(model: Any, given: str | None) -> str:
    """The id recorded for a model: `given`, else the path, else the in-memory model's name_or_path."""
    if given:
        return given
    if isinstance(model, (str, os.PathLike)):
        return str(model)
    name = getattr(model, "name_or_path", None)
    if not name:
        config = getattr(model, "config", None)
        name = getattr(config, "_name_or_path", None)
    if not name:
        raise ValueError("pass checkpoint_id / reference_id for an in-memory model")
    return str(name)


def _judge_spec(given: Any, hub_id: str | None, default: str) -> Any:
    """An explicit judge (spec, model object or False) wins; else the hub id in cfg["models"]; else the default."""
    if given is not None:
        return given
    if hub_id:
        return "hf:" + hub_id
    return default


def _recorded_judge(spec: Any, default: str) -> str | None:
    """The judge name stored in the results; None for a skipped judge."""
    if spec is False:
        return None
    return judge_name(spec, default)


def _generation_settings(cfg: Mapping[str, Any]) -> dict:
    """max_new_tokens, temperature and batch_size from cfg["generation"], with defaults."""
    gen_cfg = cfg.get("generation", {})
    return {
        "max_new_tokens": gen_cfg.get("max_new_tokens", 256),
        "temperature": gen_cfg.get("temperature", 0.0),
        "batch_size": gen_cfg.get("batch_size", 16),
    }


def _prompt_splits(selected: dict, split: str, kinds: tuple[str, ...]) -> dict[str, list]:
    """{cache split name: rows} for the selected prompts of the given kinds, e.g. {"dev_harmful_en": [...]}."""
    prompt_splits = {}
    for kind in kinds:
        for lang, rows in selected[kind].items():
            prompt_splits[_split_key(split, kind, lang)] = rows
    return prompt_splits


def _generate_splits(
    model: Any, tokenizer, splits: dict[str, list], name: str, cache: bool, root: Any, gen_settings: dict
) -> dict[str, dict[str, str]]:
    """{split: {id: completion}} for the non-empty splits, through the cache when `cache` is set."""
    splits = {split: rows for split, rows in splits.items() if rows}
    if not splits:
        return {}
    if cache:
        return generate_cached(model, tokenizer, splits, checkpoint=name, root=root, **gen_settings)

    flat_rows = []
    for split, rows in splits.items():
        for row in rows:
            flat_rows.append((split, row))
    texts = generate(model, tokenizer, [row["prompt"] for _, row in flat_rows], **gen_settings)
    completions: dict[str, dict[str, str]] = {split: {} for split in splits}
    for (split, row), text in zip(flat_rows, texts):
        completions[split][row["id"]] = text
    return completions


def _answered(selected_rows: dict[str, list], completions: dict, split: str, kind: str) -> list[_Answered]:
    """Each selected prompt of one kind with its completion, languages in order."""
    answered = []
    for lang, rows in selected_rows.items():
        texts = completions[_split_key(split, kind, lang)]
        for row in rows:
            answered.append(_Answered(lang, row, texts[row["id"]]))
    return answered


def _harm_labels(harmful: list[_Answered], harm_spec: Any, batch_size: int) -> list[dict]:
    """Llama Guard 3 labels for the harmful completions; empty labels when the judge is set to False."""
    if harm_spec is False:
        return [{"unsafe": None, "categories": []} for _ in harmful]
    prompts = [answer.row["prompt"] for answer in harmful]
    completions = [answer.completion for answer in harmful]
    return harm_judge(prompts, completions, model=harm_spec, batch_size=batch_size)


def _refusal_labels(answered: list[_Answered], refusal_spec: Any, rubric: str, batch_size: int) -> list[dict]:
    """Refusal-judge labels (REFUSE / PARTIAL / COMPLY); empty labels when the judge is set to False."""
    if refusal_spec is False:
        return [{"label": None} for _ in answered]
    return refusal_judge(
        [answer.row["prompt"] for answer in answered],
        [answer.completion for answer in answered],
        [answer.lang for answer in answered],
        model=refusal_spec,
        rubric_path=rubric,
        batch_size=batch_size,
    )


def _harmful_item(answer: _Answered, guard_label: dict, refusal_label: dict) -> dict:
    """The result entry of one harmful prompt: id, language, completion hash and judge labels (no text)."""
    return {
        "id": answer.row["id"],
        "lang": answer.lang,
        "completion_sha256": sha256(answer.completion),
        "refusal": refusal_label["label"],
        "unsafe": guard_label["unsafe"],
        "categories": guard_label["categories"],
    }


def _benign_item(answer: _Answered, refusal_label: dict) -> dict:
    """The result entry of one benign prompt: id, language, completion hash and refusal label (no text)."""
    return {
        "id": answer.row["id"],
        "lang": answer.lang,
        "completion_sha256": sha256(answer.completion),
        "refusal": refusal_label["label"],
    }


def _result_items(
    harmful: list[_Answered],
    benign: list[_Answered],
    guard_labels: list[dict],
    refusal_harmful: list[dict],
    refusal_benign: list[dict],
    teacher_forced: dict,
) -> dict:
    """Per-item results: ids, labels, scores and completion hashes only."""
    harmful_items = []
    for answer, guard_label, refusal_label in zip(harmful, guard_labels, refusal_harmful):
        harmful_items.append(_harmful_item(answer, guard_label, refusal_label))
    benign_items = []
    for answer, refusal_label in zip(benign, refusal_benign):
        benign_items.append(_benign_item(answer, refusal_label))
    mmlu_items = []
    for lang, scores in teacher_forced["mmlu"].items():
        for score in scores:
            mmlu_items.append({"lang": lang, **score})
    task_items = []
    if teacher_forced["task"]:
        task_items = teacher_forced["task"]["items"]
    return {"harmful": harmful_items, "xstest": benign_items, "mmlu": mmlu_items, "task": task_items}


def _task_summary(task: dict | None) -> dict | None:
    """The task NLL result without its per-item list (those go in "items")."""
    if not task:
        return None
    return {key: value for key, value in task.items() if key != "items"}


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
    eval_cfg = cfg.get("evaluation", {})
    models_cfg = cfg.get("models", {})
    if not judges:
        judges = {}
    selected = select_tier(data, tier, cfg)
    split = TIER_SPLIT[tier]
    checkpoint_label = _model_id(checkpoint, checkpoint_id)
    reference_label = None
    if reference is not None:
        reference_label = _model_id(reference, reference_id)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(models_cfg["aligned"])
    gen_settings = _generation_settings(cfg)

    # Step 1: generate on the harmful and benign prompts (and the reference on the benign ones, for KL).
    # A path model is loaded and unloaded inside each call.
    prompt_splits = _prompt_splits(selected, split, kinds=("harmful", "xstest"))
    completions = _generate_splits(
        checkpoint, tokenizer, prompt_splits, checkpoint_label, cache, gen_root, gen_settings
    )
    reference_answers = None
    if reference is not None:
        reference_splits = _prompt_splits(selected, split, kinds=("xstest",))
        reference_answers = _generate_splits(
            reference, tokenizer, reference_splits, reference_label, cache, gen_root, gen_settings
        )

    # Step 2: teacher-forced metrics, then release the checkpoint before any judge is loaded.
    teacher_forced = _teacher_forced(checkpoint, reference, tokenizer, selected, reference_answers, split, eval_cfg)
    free_gpu()

    # Steps 3 and 4: the judges, one at a time; each loads and unloads its own model.
    harmful = _answered(selected["harmful"], completions, split, "harmful")
    benign = _answered(selected["xstest"], completions, split, "xstest")
    harm_spec = _judge_spec(judges.get("harm"), models_cfg.get("harm_judge"), DEFAULT_HARM_JUDGE)
    refusal_spec = _judge_spec(judges.get("refusal"), models_cfg.get("refusal_judge"), DEFAULT_REFUSAL_JUDGE)
    rubric = eval_cfg.get("refusal_rubric", REFUSAL_RUBRIC)
    judge_batch_size = eval_cfg.get("judge_batch_size", 8)
    guard_labels = _harm_labels(harmful, harm_spec, judge_batch_size)  # 3: Llama Guard 3
    refusal_labels = _refusal_labels(harmful + benign, refusal_spec, rubric, judge_batch_size)  # 4: Aya Expanse
    refusal_harmful = refusal_labels[: len(harmful)]
    refusal_benign = refusal_labels[len(harmful) :]

    # Results: ids, labels and completion hashes only; the completion text is not stored anywhere below.
    items = _result_items(harmful, benign, guard_labels, refusal_harmful, refusal_benign, teacher_forced)
    langs = _ordered_langs(selected, cfg.get("languages", []))
    payload = {
        "checkpoint": checkpoint_label,
        "tier": tier,
        "data_split": split,
        "langs": langs,
        "reference": reference_label,
        "generation": {
            "max_new_tokens": gen_settings["max_new_tokens"],
            "temperature": gen_settings["temperature"],
        },
        "judges": {
            "harm": _recorded_judge(harm_spec, DEFAULT_HARM_JUDGE),
            "refusal": _recorded_judge(refusal_spec, DEFAULT_REFUSAL_JUDGE),
            "refusal_rubric_sha256": sha256(read_rubric(rubric)),
        },
        "metrics": {lang: _lang_metrics(lang, items, teacher_forced) for lang in langs},
        "task": _task_summary(teacher_forced["task"]),
        "items": items,
    }
    write_result(
        out_name, payload, config=dict(cfg), seed=cfg.get("split_seed"), checkpoint=checkpoint_label, out_dir=out_dir
    )
    return payload
