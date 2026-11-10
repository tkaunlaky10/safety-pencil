"""The fine-tune's update ΔW, read from a PEFT LoRA adapter in fp32 (CLAUDE.md rule 1).

    ΔW = scaling · lora_B @ lora_A,   scaling = lora_alpha / r   (lora_alpha / √r with rsLoRA)

Never take ΔW by subtracting bf16 merged weights: the rounding error is comparable to the update.
Module names are returned in the base model's naming ("model.layers.3.self_attn.q_proj"), which is
what the factor files and the repair use.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

# "base_model.model.<module>.lora_A[.<adapter>].weight" -> module, "A" or "B"
_LORA_KEY = re.compile(r"^(?:base_model\.model\.)?(?P<mod>.+?)\.lora_(?P<ab>[AB])(?:\.(?P<adapter>[^.]+))?\.weight$")


def _scaling(cfg: dict, module: str) -> float:
    """lora_alpha / r (or / √r with rsLoRA), honouring per-module rank and alpha patterns."""
    rank = _pattern_override(cfg.get("rank_pattern"), module, default=cfg["r"])
    alpha = _pattern_override(cfg.get("alpha_pattern"), module, default=cfg["lora_alpha"])
    if cfg.get("use_rslora"):
        return alpha / math.sqrt(rank)
    return alpha / rank


def _pattern_override(patterns: dict | None, module: str, default):
    """The value of the last regex in {pattern: value} that matches the module name, else `default`."""
    value = default
    if not patterns:
        return value
    for pattern, override in patterns.items():
        if re.search(pattern, module):
            value = override
    return value


@dataclass
class LoRADelta:
    """ΔW = B @ A kept low-rank (scaling folded into B, float32): a whole 36-layer delta costs megabytes
    instead of ~11 GB. Use `as_dense` where a module's full matrix is needed."""

    B: torch.Tensor
    A: torch.Tensor

    @property
    def shape(self):
        return torch.Size((self.B.shape[0], self.A.shape[1]))

    def dense(self, dtype=torch.float32, device=None) -> torch.Tensor:
        """B @ A as a dense matrix (on B's device by default)."""
        if device is None:
            device = self.B.device
        return self.B.to(device, dtype) @ self.A.to(device, dtype)


def as_dense(delta, dtype=torch.float32, device=None) -> torch.Tensor:
    """A delta (dense tensor or LoRADelta) as a dense tensor of the given dtype and device (default: its own)."""
    if isinstance(delta, LoRADelta):
        return delta.dense(dtype, device)
    if device is None:
        device = delta.device
    return delta.to(device, dtype)


def lora_delta(
    adapter_path: str | Path, device: str | torch.device = "cpu", lowrank: bool = False
) -> dict[str, torch.Tensor]:
    """{module: ΔW (d_out × d_in, float32)} from a PEFT adapter folder; LoRADelta objects with lowrank=True
    (use that for the real model)."""
    from safetensors.torch import load_file

    path = Path(adapter_path)
    cfg = _read_lora_config(path)
    state_dict = load_file(str(path / "adapter_model.safetensors"), device=str(device))
    deltas = {}
    for module, lora_weights in _lora_weights_by_module(state_dict).items():
        if set(lora_weights) != {"A", "B"}:
            raise ValueError(f"{module}: missing lora_A or lora_B")
        lora_B = _scaling(cfg, module) * lora_weights["B"].float()
        lora_A = lora_weights["A"].float()
        if lowrank:
            deltas[module] = LoRADelta(lora_B, lora_A)
        else:
            deltas[module] = lora_B @ lora_A
    if not deltas:
        raise ValueError(f"no LoRA weights found in {path}")
    return deltas


def _read_lora_config(path: Path) -> dict:
    """adapter_config.json, checked to be a plain LoRA adapter."""
    cfg = json.loads((path / "adapter_config.json").read_text())
    if cfg.get("use_dora"):
        raise NotImplementedError("DoRA adapters are not a plain B@A update")
    if cfg.get("peft_type", "LORA") != "LORA":
        raise ValueError(f"not a LoRA adapter: {cfg.get('peft_type')}")
    return cfg


def _lora_weights_by_module(state_dict: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """{module: {"A": lora_A, "B": lora_B}} from an adapter's state dict (other keys are ignored)."""
    by_module: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in state_dict.items():
        match = _LORA_KEY.match(key)
        if match:
            by_module.setdefault(match["mod"], {})[match["ab"]] = tensor
    return by_module


def merge_deltas(model: nn.Module, deltas: dict[str, torch.Tensor], scale: float = 1.0) -> nn.Module:
    """W ← W + scale·ΔW in place, computed in fp32 and cast back to W's dtype. Returns the model."""
    with torch.no_grad():
        for name, delta in deltas.items():
            W = model.get_submodule(name).weight
            if W.shape != delta.shape:
                raise ValueError(f"{name}: delta {tuple(delta.shape)} vs weight {tuple(W.shape)}")
            W.copy_((W.float() + scale * as_dense(delta, torch.float32, W.device)).to(W.dtype))
    return model


def set_weights(model: nn.Module, weights: dict[str, torch.Tensor]) -> nn.Module:
    """Overwrite module weights (e.g. repaired W0 + ΔW′) in place."""
    with torch.no_grad():
        for name, W in weights.items():
            target = model.get_submodule(name).weight
            target.copy_(W.to(target))
    return model


def model_weights(model: nn.Module, names, device=None) -> dict[str, torch.Tensor]:
    """{module: weight} copies for the named linear modules (on `device`, default the weight's own)."""
    weights = {}
    for name in names:
        W = model.get_submodule(name).weight
        target_device = W.device if device is None else device
        weights[name] = W.detach().to(target_device, copy=True)
    return weights


def save_checkpoint(model: nn.Module, tokenizer, path: str | Path, note: str) -> Path:
    """Save weights (or a PEFT adapter) with the pinned tokenizer, so no checkpoint can bring a different chat
    template, and a RESEARCH_ARTIFACT.txt label (CLAUDE.md rule 7)."""
    path = Path(path)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    (path / "RESEARCH_ARTIFACT.txt").write_text(note + "\n")
    return path
