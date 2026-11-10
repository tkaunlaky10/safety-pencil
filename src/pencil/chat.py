"""Chat templating with the positions the method needs.

Every conversation goes through the model's own chat template (Tiny Aya always injects a system
preamble), and we record where the user's text starts and where the response starts:

    [ system preamble | user turn ......... | assistant response ...... ]
                      ^ user_start          ^ response_start

- Loss positions are the positions whose next-token prediction is a response token:
  response_start - 1 ... len - 2.
- The "all" row set is every position from user_start on: the user prompt and the response, without
  the preamble, which is identical in every example.
- The "full" row set is every position, preamble included. The preamble's hidden states are identical in
  every example (causal attention), but k_proj and v_proj at preamble positions feed every later token,
  so leaving those rows out misses most of the k/v curvature (see tests/test_model_repair.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _ids(x) -> list[int]:
    """apply_chat_template returns a list or a BatchEncoding depending on the transformers version."""
    if isinstance(x, dict) or hasattr(x, "keys"):
        x = x["input_ids"]
    if isinstance(x, torch.Tensor):
        x = x.tolist()
    if x and isinstance(x[0], list):
        x = x[0]
    return list(x)


def template_ids(tok, messages, *, add_generation_prompt=False) -> list[int]:
    return _ids(tok.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=True))


def user_content_offset(tok, system: str | None = None) -> int:
    """Number of template tokens before the user's text: the common prefix of two different prompts."""

    def msgs(u):
        return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": u}]

    a = template_ids(tok, msgs("alpha beta"))
    b = template_ids(tok, msgs("zulu yankee"))
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def prompt_suffix_len(tok, system: str | None = None) -> int:
    """Number of template tokens after the user's text in a generation prompt (end-of-turn, start of the
    assistant turn): the common suffix of two different prompts."""

    def msgs(u):
        return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": u}]

    a = template_ids(tok, msgs("alpha beta"), add_generation_prompt=True)
    b = template_ids(tok, msgs("zulu yankee"), add_generation_prompt=True)
    n = 0
    while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n


@dataclass
class Encoded:
    input_ids: list[int]
    user_start: int
    response_start: int  # == len(input_ids) when there is no response

    @property
    def loss_positions(self) -> range:
        return range(self.response_start - 1, len(self.input_ids) - 1)


def encode_chat(
    tok,
    prompt: str,
    response: str | None = None,
    *,
    system: str | None = None,
    max_len: int | None = None,
    user_offset: int | None = None,
    suffix_len: int | None = None,
) -> Encoded:
    """Template one (prompt, response) pair. If max_len truncates, the end of the user's text is cut; the
    template's turn markers and the response are always kept."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    prompt_ids = template_ids(tok, msgs, add_generation_prompt=True)
    if response is None:
        full = prompt_ids
    else:
        full = template_ids(tok, msgs + [{"role": "assistant", "content": response}])
        if full[: len(prompt_ids)] != prompt_ids:
            raise ValueError("chat template is not prefix-stable; cannot locate the response")
    user_start = user_offset if user_offset is not None else user_content_offset(tok, system)
    resp_start = len(prompt_ids)
    if max_len is not None and len(full) > max_len:
        cut = len(full) - max_len  # drop tokens from the end of the user text
        suf = suffix_len if suffix_len is not None else prompt_suffix_len(tok, system)
        user_end = resp_start - suf  # the generation-prompt markers start here
        if user_end - cut <= user_start:
            raise ValueError(f"max_len={max_len} too short for this example")
        full = full[: user_end - cut] + full[user_end:]
        resp_start -= cut
    return Encoded(full, user_start, resp_start)


def collate(encoded: list[Encoded], pad_id: int, *, side: str = "right") -> dict[str, torch.Tensor]:
    """Pad to a batch and build masks: attention_mask, loss_mask (scored positions), rows_all and rows_full."""
    T = max(len(e.input_ids) for e in encoded)
    B = len(encoded)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    att = torch.zeros(B, T, dtype=torch.long)
    loss = torch.zeros(B, T, dtype=torch.bool)
    rows_all = torch.zeros(B, T, dtype=torch.bool)
    rows_full = torch.zeros(B, T, dtype=torch.bool)
    for b, e in enumerate(encoded):
        n = len(e.input_ids)
        off = 0 if side == "right" else T - n
        ids[b, off : off + n] = torch.tensor(e.input_ids)
        att[b, off : off + n] = 1
        for t in e.loss_positions:
            loss[b, off + t] = True
        rows_all[b, off + e.user_start : off + n - 1] = True  # last position predicts nothing
        rows_full[b, off : off + n - 1] = True
    return {"input_ids": ids, "attention_mask": att, "loss_mask": loss, "rows_all": rows_all, "rows_full": rows_full}
