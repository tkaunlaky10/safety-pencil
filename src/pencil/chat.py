"""Chat templating with the token positions the method needs.

Every conversation goes through the model's own chat template. Tiny Aya's template always injects a
system preamble of about 366 tokens, identical in every example. We record where the user's text starts
(``user_start``) and where the response starts (``response_start``), and ``collate`` turns those two
numbers into the masks for the factor row sets. One character per token (p = preamble, u = user text,
m = turn markers, r = response, L = sequence length):

    position       0         user_start        response_start     L-1
                   v         v                 v                  v
    token          ppppppppppuuuuuuuuuuuuuummmmrrrrrrrrrrrrrrrrrrrr
    loss_mask                                 xxxxxxxxxxxxxxxxxxxx.     response_start-1 .. L-2
    rows_all                 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.     user_start .. L-2
    rows_full      xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.     0 .. L-2

The logits at position t predict token t + 1, so the last position (L-1, ".") is in no row set, and the
loss starts one position before the response. The row sets of ``pencil.factors`` read these masks:

- ``loss``: ``loss_mask``, the positions whose prediction is a response token.
- ``all``: ``rows_all``, the user prompt and the response, without the preamble.
- ``full``: ``rows_full``, every position, preamble included. The preamble's hidden states are identical in
  every example (causal attention), but k_proj and v_proj at preamble positions feed every later token,
  so leaving those rows out misses most of the k/v curvature (see tests/test_model_repair.py).
- ``full_reduce``: also ``rows_full``, reduced to one (mean input, summed gradient) pair per sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Two different user texts: the template tokens they share at the start (end) are the template's own.
_PROBE_TEXTS = ("alpha beta", "zulu yankee")


# ══ Templating ═══════════════════════════════════════════════════════════════


def _as_id_list(encoded) -> list[int]:
    """Token ids as a flat list: apply_chat_template returns a list, a nested list, a tensor or a
    BatchEncoding depending on the transformers version."""
    if hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return list(encoded)


def template_ids(tokenizer, messages, *, add_generation_prompt=False) -> list[int]:
    """Token ids of ``messages`` rendered with the tokenizer's chat template."""
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=True)
    return _as_id_list(encoded)


def _conversation(user_text: str, system: str | None) -> list[dict[str, str]]:
    """One user turn, after a system turn when ``system`` is given."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_text})
    return messages


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    """Number of leading tokens that a and b share."""
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def user_content_offset(tokenizer, system: str | None = None) -> int:
    """Number of template tokens before the user's text: the common prefix of two different prompts."""
    first = template_ids(tokenizer, _conversation(_PROBE_TEXTS[0], system))
    second = template_ids(tokenizer, _conversation(_PROBE_TEXTS[1], system))
    return _common_prefix_len(first, second)


def prompt_suffix_len(tokenizer, system: str | None = None) -> int:
    """Number of template tokens after the user's text in a generation prompt (end-of-turn, start of the
    assistant turn): the common suffix of two different prompts."""
    first = template_ids(tokenizer, _conversation(_PROBE_TEXTS[0], system), add_generation_prompt=True)
    second = template_ids(tokenizer, _conversation(_PROBE_TEXTS[1], system), add_generation_prompt=True)
    return _common_prefix_len(first[::-1], second[::-1])


# ══ Encoding one example ═════════════════════════════════════════════════════


@dataclass
class Encoded:
    """One templated example: its token ids and where the user text and the response start."""

    input_ids: list[int]
    user_start: int
    response_start: int  # == len(input_ids) when there is no response

    @property
    def loss_positions(self) -> range:
        """Positions whose next-token prediction is a response token."""
        return range(self.response_start - 1, len(self.input_ids) - 1)


def encode_chat(
    tokenizer,
    prompt: str,
    response: str | None = None,
    *,
    system: str | None = None,
    max_len: int | None = None,
    user_offset: int | None = None,
    suffix_len: int | None = None,
) -> Encoded:
    """Template one (prompt, response) pair. If max_len truncates, the end of the user's text is cut; the
    template's turn markers and the response are always kept.

    ``user_offset`` and ``suffix_len`` are the template's constants (``user_content_offset``,
    ``prompt_suffix_len``); pass them in to avoid recomputing them for every example.
    """
    messages = _conversation(prompt, system)
    prompt_ids = template_ids(tokenizer, messages, add_generation_prompt=True)
    if response is None:
        input_ids = prompt_ids
    else:
        input_ids = template_ids(tokenizer, messages + [{"role": "assistant", "content": response}])
        if input_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("chat template is not prefix-stable; cannot locate the response")

    if user_offset is None:
        user_offset = user_content_offset(tokenizer, system)
    response_start = len(prompt_ids)
    if max_len is not None and len(input_ids) > max_len:
        if suffix_len is None:
            suffix_len = prompt_suffix_len(tokenizer, system)
        input_ids, response_start = _cut_end_of_user_text(input_ids, user_offset, response_start, suffix_len, max_len)
    return Encoded(input_ids, user_offset, response_start)


def _cut_end_of_user_text(
    input_ids: list[int], user_start: int, response_start: int, suffix_len: int, max_len: int
) -> tuple[list[int], int]:
    """Drop the last tokens of the user's text so the example fits in max_len; returns the shortened ids and
    the new response start.

        before  [ preamble | user text .............. | turn marks | response ]
        after   [ preamble | user text ...... | turn marks | response ]
                                               ^ user_end - n_cut
    """
    n_cut = len(input_ids) - max_len
    user_end = response_start - suffix_len  # the generation-prompt markers start here
    if user_end - n_cut <= user_start:
        raise ValueError(f"max_len={max_len} too short for this example")
    shortened = input_ids[: user_end - n_cut] + input_ids[user_end:]
    return shortened, response_start - n_cut


# ══ Batching and row-set masks ═══════════════════════════════════════════════


def collate(encoded: list[Encoded], pad_id: int, *, side: str = "right") -> dict[str, torch.Tensor]:
    """Pad to a batch and build masks: attention_mask, loss_mask (scored positions), rows_all and rows_full.

    The masks follow the layout in the module docstring, shifted right by the padding when ``side`` is
    "left" (any value other than "right" pads on the left).
    """
    batch_size = len(encoded)
    padded_len = max(len(example.input_ids) for example in encoded)
    input_ids = torch.full((batch_size, padded_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, padded_len, dtype=torch.long)
    loss_mask = torch.zeros(batch_size, padded_len, dtype=torch.bool)
    rows_all = torch.zeros(batch_size, padded_len, dtype=torch.bool)
    rows_full = torch.zeros(batch_size, padded_len, dtype=torch.bool)

    for row, example in enumerate(encoded):
        length = len(example.input_ids)
        start = 0 if side == "right" else padded_len - length
        end = start + length
        input_ids[row, start:end] = torch.tensor(example.input_ids)
        attention_mask[row, start:end] = 1
        for position in example.loss_positions:
            loss_mask[row, start + position] = True
        # The last position predicts nothing, so every row set stops one short of the end.
        rows_all[row, start + example.user_start : end - 1] = True
        rows_full[row, start : end - 1] = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "rows_all": rows_all,
        "rows_full": rows_full,
    }
