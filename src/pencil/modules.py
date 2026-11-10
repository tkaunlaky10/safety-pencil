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

_MODULE_NAME = re.compile(r"^(?P<prefix>.*?)layers\.(?P<layer>\d+)\.(?:self_attn|mlp)\.(?P<leaf>\w+_proj)$")
_LAYER_INDEX = re.compile(r"layers\.(\d+)\.")


def parse_name(name: str) -> tuple[str, int, str]:
    """Split a decoder linear module's name: 'model.layers.3.self_attn.q_proj' -> ('model.', 3, 'q_proj')."""
    match = _MODULE_NAME.match(name)
    if not match:
        raise ValueError(f"not a decoder linear module: {name}")
    return match["prefix"], int(match["layer"]), match["leaf"]


def leaf(name: str) -> str:
    """The last part of a module name: 'model.layers.3.self_attn.q_proj' -> 'q_proj'."""
    return name.rsplit(".", 1)[-1]


def layer_of(name: str) -> int:
    """Layer index in any key that contains 'layers.<L>.' (-1 if none)."""
    match = _LAYER_INDEX.search(name)
    if not match:
        return -1
    return int(match.group(1))


def layer_order(names: Iterable[str]) -> list[str]:
    """Sort by layer, then name, so per-layer data is loaded once when iterating."""
    return sorted(names, key=_layer_then_name)


def _layer_then_name(name: str) -> tuple[int, str]:
    return layer_of(name), name


def target_modules(model: nn.Module, layers: Iterable[int] | None = None) -> dict[str, nn.Linear]:
    """Every q/k/v/o/gate/up/down nn.Linear in the decoder (optionally only in `layers`)."""
    wanted_layers = None
    if layers is not None:
        wanted_layers = {int(layer) for layer in layers}
    targets = {}
    for name, module in model.named_modules():
        if _is_target(name, module, wanted_layers):
            targets[name] = module
    if not targets:
        raise ValueError("no plain nn.Linear decoder modules found; merge any PEFT adapter into the weights first")
    return targets


def _is_target(name: str, module: nn.Module, wanted_layers: set[int] | None) -> bool:
    """True for a plain nn.Linear q/k/v/o/gate/up/down module in one of the wanted layers (None: any)."""
    match = _MODULE_NAME.match(name)
    if not match or match["leaf"] not in LEAVES or not isinstance(module, nn.Linear):
        return False
    return wanted_layers is None or int(match["layer"]) in wanted_layers


def is_parallel_block(model: nn.Module) -> bool:
    """True for model types whose attention and MLP read the same input_layernorm output."""
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    return model_type in PARALLEL_MODEL_TYPES


def input_factor_key(name: str, shared: bool) -> str:
    """Key of the input factor a module uses: the layer's shared key for q/k/v/gate/up in a parallel block,
    otherwise the module's own name."""
    prefix, layer, module_leaf = parse_name(name)
    if shared and module_leaf in SHARED_INPUT:
        return f"{prefix}layers.{layer}.input"
    return name


def decoder(model: nn.Module) -> nn.Module:
    """The transformer body (embeddings + layers + final norm) of a causal LM."""
    body = getattr(model, "model", None)
    if body is None:
        return model.get_decoder()
    return body


def lm_logits(model: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """LM head on chosen hidden states only (the vocabulary is 262k, so never run it on every position)."""
    logits = model.get_output_embeddings()(hidden)
    scale = getattr(model, "logit_scale", None)  # Cohere multiplies logits by logit_scale
    if scale is None:
        return logits
    return logits * scale
