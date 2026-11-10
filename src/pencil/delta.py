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
from pathlib import Path

import torch
from torch import nn

_KEY = re.compile(r"^(?:base_model\.model\.)?(?P<mod>.+?)\.lora_(?P<ab>[AB])(?:\.(?P<adapter>[^.]+))?\.weight$")


def adapter_config(path: str | Path) -> dict:
    return json.loads((Path(path) / "adapter_config.json").read_text())


def _scaling(cfg: dict, module: str) -> float:
    r = cfg["r"]
    alpha = cfg["lora_alpha"]
    for pat, v in (cfg.get("rank_pattern") or {}).items():
        if re.search(pat, module):
            r = v
    for pat, v in (cfg.get("alpha_pattern") or {}).items():
        if re.search(pat, module):
            alpha = v
    return alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r


class LoRADelta:
    """ΔW = B @ A kept in low-rank form (scaling folded into B, float32). Behaves like the dense tensor where
    the pipeline needs one (`.to`, `.double`, `.float`, `.shape`, `.norm`, `.dense()`), materialising it only
    per module, so a 36-layer delta costs megabytes instead of ~11 GB."""

    def __init__(self, B: torch.Tensor, A: torch.Tensor):
        self.B, self.A = B, A

    @property
    def shape(self):
        return torch.Size((self.B.shape[0], self.A.shape[1]))

    @property
    def dtype(self):
        return self.B.dtype

    @property
    def device(self):
        return self.B.device

    def dense(self, dtype=torch.float32, device=None) -> torch.Tensor:
        dev = device or self.B.device
        return self.B.to(dev, dtype) @ self.A.to(dev, dtype)

    def to(self, *args, **kw) -> torch.Tensor:
        """Dense, like Tensor.to(device, dtype)."""
        dense = self.dense(torch.float64 if torch.float64 in args or kw.get("dtype") == torch.float64 else torch.float32)
        return dense.to(*args, **kw)

    def double(self) -> torch.Tensor:
        return self.dense(torch.float64)

    def float(self) -> torch.Tensor:
        return self.dense(torch.float32)

    def norm(self) -> torch.Tensor:
        B, A = self.B.double(), self.A.double()
        return torch.trace((B.T @ B) @ (A @ A.T)).clamp_min(0).sqrt()

    def __add__(self, other):
        return self.dense(other.dtype if isinstance(other, torch.Tensor) else torch.float32) + other

    __radd__ = __add__


def as_dense(d, dtype=torch.float32, device=None) -> torch.Tensor:
    return d.dense(dtype, device) if isinstance(d, LoRADelta) else d.to(device or d.device, dtype)


def lora_delta(adapter_path: str | Path, device: str | torch.device = "cpu",
               lowrank: bool = False) -> dict[str, torch.Tensor]:
    """{module_name: ΔW (d_out × d_in, float32)} from adapter_config.json + adapter_model.safetensors.
    lowrank=True returns LoRADelta objects instead (use it for full-size models)."""
    from safetensors.torch import load_file

    path = Path(adapter_path)
    cfg = adapter_config(path)
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
    mods = dict(model.named_modules())
    with torch.no_grad():
        for name, d in deltas.items():
            if name not in mods:
                raise KeyError(f"{name} is not a module of this model")
            W = mods[name].weight
            if W.shape != d.shape:
                raise ValueError(f"{name}: delta {tuple(d.shape)} vs weight {tuple(W.shape)}")
            W.copy_((W.float() + scale * as_dense(d, torch.float32, W.device)).to(W.dtype))
    return model


def set_weights(model: nn.Module, weights: dict[str, torch.Tensor]) -> nn.Module:
    """Overwrite module weights (e.g. repaired W0 + ΔW′) in place."""
    mods = dict(model.named_modules())
    with torch.no_grad():
        for name, W in weights.items():
            mods[name].weight.copy_(W.to(mods[name].weight))
    return model


def merge_to_checkpoint(base: str | nn.Module, deltas: dict[str, torch.Tensor], out_path: str | Path, *,
                        tokenizer=None, dtype=torch.bfloat16, note: str | None = None, **load_kw) -> Path:
    """Load (or take) the base model, merge the deltas in fp32, save merged weights + tokenizer.

    `note` is written to RESEARCH_ARTIFACT.txt next to the weights; use it for the CLAUDE.md rule 7 label
    ("research artifact: unrepaired, do not deploy") on drifted checkpoints."""
    if isinstance(base, (str, Path)):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(str(base), dtype=dtype, **load_kw)
    else:
        model = base
    merge_deltas(model, deltas)
    out = Path(out_path)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    if tokenizer is not None:
        tokenizer.save_pretrained(out)
    if note:
        (out / "RESEARCH_ARTIFACT.txt").write_text(note + "\n")
    return out


def delta_norms(deltas: dict[str, torch.Tensor]) -> dict[str, float]:
    return {k: float(v.norm()) for k, v in deltas.items()}
