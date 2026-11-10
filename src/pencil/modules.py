"""Names and structure of the decoder's linear layers, shared by every part of the pipeline.

A module name looks like "model.layers.3.self_attn.q_proj"; its "leaf" is "q_proj". Tiny Aya (cohere2)
has a parallel block: attention and MLP both read the output of `input_layernorm`, so q, k, v, gate and up
see the same input and share one input factor, stored under "model.layers.3.input".
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import torch
from torch import nn

LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SHARED_INPUT = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")  # read input_layernorm's output
RESIDUAL_WRITERS = ("o_proj", "down_proj")  # write into the residual stream
MLP_WIDE_OUTPUT = ("gate_proj", "up_proj")  # output side is intermediate_size
MLP_WIDE_INPUT = ("down_proj",)  # input side is intermediate_size
PARALLEL_MODEL_TYPES = ("cohere", "cohere2")

_NAME = re.compile(r"^(?P<prefix>.*?)layers\.(?P<layer>\d+)\.(?:self_attn|mlp)\.(?P<leaf>\w+_proj)$")
_LAYER = re.compile(r"layers\.(\d+)\.")


def parse_name(name: str) -> tuple[str, int, str]:
    """'model.layers.3.self_attn.q_proj' -> ('model.', 3, 'q_proj')."""
    m = _NAME.match(name)
    if not m:
        raise ValueError(f"not a decoder linear module: {name}")
    return m["prefix"], int(m["layer"]), m["leaf"]


def leaf(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def layer_of(name: str) -> int:
    """Layer index in any key that contains 'layers.<L>.' (-1 if none)."""
    m = _LAYER.search(name)
    return int(m.group(1)) if m else -1


def layer_order(names: Iterable[str]) -> list[str]:
    """Sort by layer, then name, so per-layer data is loaded once when iterating."""
    return sorted(names, key=lambda n: (layer_of(n), n))


def target_modules(model: nn.Module, layers: Iterable[int] | None = None) -> dict[str, nn.Linear]:
    """Every q/k/v/o/gate/up/down nn.Linear in the decoder (optionally only in `layers`)."""
    keep = None if layers is None else {int(x) for x in layers}
    out = {}
    for name, module in model.named_modules():
        m = _NAME.match(name)
        if m and m["leaf"] in LEAVES and isinstance(module, nn.Linear) and (keep is None or int(m["layer"]) in keep):
            out[name] = module
    if not out:
        raise ValueError("no plain nn.Linear decoder modules found; merge any PEFT adapter into the weights first")
    return out


def is_parallel_block(model: nn.Module) -> bool:
    return getattr(getattr(model, "config", None), "model_type", None) in PARALLEL_MODEL_TYPES


def input_factor_key(name: str, shared: bool) -> str:
    """Key of the input factor a module uses: the layer's shared key for q/k/v/gate/up in a parallel block,
    otherwise the module's own name."""
    prefix, layer, lf = parse_name(name)
    return f"{prefix}layers.{layer}.input" if shared and lf in SHARED_INPUT else name


def decoder(model: nn.Module) -> nn.Module:
    """The transformer body (embeddings + layers + final norm) of a causal LM."""
    body = getattr(model, "model", None)
    return body if body is not None else model.get_decoder()


def lm_logits(model: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """LM head on chosen hidden states only (the vocabulary is 262k, so never run it on every position)."""
    logits = model.get_output_embeddings()(hidden)
    scale = getattr(model, "logit_scale", None)  # Cohere multiplies logits by logit_scale
    return logits * scale if scale is not None else logits
