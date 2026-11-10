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

_KEY = re.compile(r"^(?:base_model\.model\.)?(?P<mod>.+?)\.lora_(?P<ab>[AB])(?:\.(?P<adapter>[^.]+))?\.weight$")


def _scaling(cfg: dict, module: str) -> float:
    """lora_alpha / r (or / √r with rsLoRA), honouring per-module rank and alpha patterns."""
    r = cfg["r"]
    alpha = cfg["lora_alpha"]
    for pat, v in (cfg.get("rank_pattern") or {}).items():
        if re.search(pat, module):
            r = v
    for pat, v in (cfg.get("alpha_pattern") or {}).items():
        if re.search(pat, module):
            alpha = v
    return alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r


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
        device = device or self.B.device
        return self.B.to(device, dtype) @ self.A.to(device, dtype)


def as_dense(d, dtype=torch.float32, device=None) -> torch.Tensor:
    """A delta (dense tensor or LoRADelta) as a dense tensor of the given dtype and device."""
    return d.dense(dtype, device) if isinstance(d, LoRADelta) else d.to(device or d.device, dtype)


def lora_delta(
    adapter_path: str | Path, device: str | torch.device = "cpu", lowrank: bool = False
) -> dict[str, torch.Tensor]:
    """{module: ΔW (d_out × d_in, float32)} from a PEFT adapter folder; LoRADelta objects with lowrank=True
    (use that for the real model)."""
    from safetensors.torch import load_file

    path = Path(adapter_path)
    cfg = json.loads((path / "adapter_config.json").read_text())
    if cfg.get("use_dora"):
        raise NotImplementedError("DoRA adapters are not a plain B@A update")
    if cfg.get("peft_type", "LORA") != "LORA":
        raise ValueError(f"not a LoRA adapter: {cfg.get('peft_type')}")
    sd = load_file(str(path / "adapter_model.safetensors"), device=str(device))
    parts: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in sd.items():
        m = _KEY.match(k)
        if m:
            parts.setdefault(m["mod"], {})[m["ab"]] = v
    out = {}
    for mod, ab in parts.items():
        if set(ab) != {"A", "B"}:
            raise ValueError(f"{mod}: missing lora_A or lora_B")
        B = _scaling(cfg, mod) * ab["B"].float()
        out[mod] = LoRADelta(B, ab["A"].float()) if lowrank else B @ ab["A"].float()
    if not out:
        raise ValueError(f"no LoRA weights found in {path}")
    return out


def merge_deltas(model: nn.Module, deltas: dict[str, torch.Tensor], scale: float = 1.0) -> nn.Module:
    """W ← W + scale·ΔW in place, computed in fp32 and cast back to W's dtype. Returns the model."""
    with torch.no_grad():
        for name, d in deltas.items():
            W = model.get_submodule(name).weight
            if W.shape != d.shape:
                raise ValueError(f"{name}: delta {tuple(d.shape)} vs weight {tuple(W.shape)}")
            W.copy_((W.float() + scale * as_dense(d, torch.float32, W.device)).to(W.dtype))
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
    return {
        n: model.get_submodule(n).weight.detach().to(device or model.get_submodule(n).weight.device, copy=True)
        for n in names
    }


def save_checkpoint(model: nn.Module, tokenizer, path: str | Path, note: str) -> Path:
    """Save weights (or a PEFT adapter) with the pinned tokenizer, so no checkpoint can bring a different chat
    template, and a RESEARCH_ARTIFACT.txt label (CLAUDE.md rule 7)."""
    path = Path(path)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    (path / "RESEARCH_ARTIFACT.txt").write_text(note + "\n")
    return path
